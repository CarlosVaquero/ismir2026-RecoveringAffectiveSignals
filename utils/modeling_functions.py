# ==============================================
# GLOBAL PER-MODULE PCA FIT 
# with optional global decorrelation stage
# ==============================================

from utils.group_musif_features import module_regex_map
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dataclasses import dataclass, field


# -------
@dataclass
class PCAConfig:
    """Configuration for PCA reduction pipeline."""
    # Modes:
    #   - "modular": per-module PCA with caps (base caps + modular_extra_caps)
    #   - "modular_uncapped": per-module PCA with NO caps
    #   - "global": single global PCA on raw features, then PCs attributed to modules
    mode: str = "global"                           # "modular" | "modular_uncapped" | "global"
    var_ratio_per_module: float = 0.90             # target variance for per-module PCA
    var_ratio_global: float = 0.90                 # target variance for global PCA
    keep_singletons: bool = False                  # pass-through modules with <2 features
    musical_features: Optional[List[str]] = None   # filter modules; None = all

    # Caps configuration (only used when mode == "modular")
    module_caps: Dict[str, int] = field(default_factory=lambda: {
        "texture": 40,
        "melody":  35,
        "scale":   25,
        "unknown": 30,
    })
    modular_extra_caps: Dict[str, int] = field(default_factory=lambda: {
        "harmony": 30,
        "rhythm":  35,
    })

    def __post_init__(self):
        if self.mode not in ["modular", "modular_uncapped", "global"]:
            raise ValueError(f"Invalid mode: {self.mode}")

    def get_effective_caps(self) -> Dict[str, int]:
        """
        Return caps to apply based on reduction mode.
        - modular:          apply module_caps + modular_extra_caps
        - modular_uncapped: no caps at all (empty dict)
        - global:           caps irrelevant (empty dict)
        """
        if self.mode == "modular":
            caps = {**self.module_caps}
            caps.update(self.modular_extra_caps)
            return caps
        return {}


class PCAReductionPipeline:
    """Two-stage PCA reduction: per-module (Stage 1) → optional global (Stage 2).

    Modes
    -----
    modular / modular_uncapped
        Stage 1: per-module PCA (with or without caps).
        df_stage1 = MultiIndex (module, PCk) per-module result.
        df_final  = same as df_stage1 (flat).

    global
        No per-module stage. One PCA is fit on ALL raw features at once.
        Each resulting PC is then attributed to the module whose features
        carry the largest share of that PC's squared loadings.
        df_stage1 = MultiIndex (module, PCk) with attributed PCs.
        df_final  = flat DataFrame with columns "global__PC1", "global__PC2", …
    """

    def __init__(self, config: PCAConfig):
        self.config = config
        self.scalers: Dict[str, StandardScaler] = {}
        self.pcas: Dict[str, PCA] = {}
        self.module_dataframes: Dict[str, pd.DataFrame] = {}
        self.summary_rows: List[Dict] = []
        self.global_scaler = None
        self.global_pca = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit the PCA pipeline on input dataframe.

        Returns
        -------
        df_stage1 : Module-segmented result.
            · modular / modular_uncapped : per-module PCA outputs (MultiIndex).
            · global                     : global PCA outputs re-attributed to
                                           modules by dominant loading (MultiIndex).
        df_final : Final flat result.
            · modular / modular_uncapped : flat version of df_stage1.
            · global                     : flat DataFrame "global__PC1", …
        """
        df = df.drop(columns=['Id'], errors='ignore')

        if self.config.mode == "global":
            df_stage1, df_final = self._global_pca_with_module_segmentation(df)
        else:
            self._stage1_per_module_pca(df)
            df_stage1 = self._get_stage1_dataframe()
            df_final  = df_stage1

        return df_stage1, df_final

    def print_summary(self) -> None:
        """Print reduction summary."""
        summary_df = (
            pd.DataFrame(self.summary_rows)
            .sort_values("module")
            .reset_index(drop=True)
        )
        print(f"\n{'='*60}")
        print(f"PCA Reduction Summary")
        print(f"{'='*60}")
        print(f"Mode: {self.config.mode}")
        if self.config.mode == "global":
            print(f"Variance ratio (global): {self.config.var_ratio_global}")
            print("(Module columns show PCs attributed by dominant loading)")
        else:
            print(f"Variance ratio (per-module): {self.config.var_ratio_per_module}")
        print(f"\n{summary_df.to_string(index=False)}")
        if self.config.mode == "modular":
            print(f"\nModule caps applied: {self.config.get_effective_caps()}")

    def get_summary_df(self) -> pd.DataFrame:
        """Return summary as dataframe."""
        return pd.DataFrame(self.summary_rows).sort_values("module").reset_index(drop=True)

    def get_flat_stage1(self) -> pd.DataFrame:
        """
        Return the stage-1 result as a flat DataFrame with columns
        named '{module}__{pc}' — consistent across all modes.

        Use this as the feature matrix for downstream analysis instead
        of df_final, which in global mode uses generic 'global__PCk' names
        that lose module attribution.
        """
        return self._get_stage1_dataframe(flatten=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Global mode
    # ─────────────────────────────────────────────────────────────────────────

    def _global_pca_with_module_segmentation(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Global mode pipeline:
          1. Run a single PCA on ALL raw features (no per-module split).
          2. Attribute each resulting PC to the module whose features carry the
             largest share of that PC's squared loadings.
          3. Populate module_dataframes from those attributions so the rest of
             the pipeline (summaries, downstream consumers) works identically
             to the modular modes.

        Returns
        -------
        df_stage1 : MultiIndex DataFrame — (module, PCk) columns.
        df_final  : Flat DataFrame — columns "global__PC1", "global__PC2", …
        """
        feature_cols = list(df.columns)

        # 1. Fit global scaler + PCA on all raw features
        X = df.values
        self.global_scaler = StandardScaler()
        X_scaled = self.global_scaler.fit_transform(X)

        self.global_pca = PCA(
            n_components=self.config.var_ratio_global, svd_solver="full"
        )
        X_pca = self.global_pca.fit_transform(X_scaled)   # (n_samples, n_global_pcs)

        explained_var = float(np.sum(self.global_pca.explained_variance_ratio_))
        n_global = X_pca.shape[1]
        print(f"Global PCA: {len(feature_cols):3d} → {n_global:3d} "
              f"({explained_var:6.1%} variance)")

        # 2. Map each original feature → its module
        grouped_features = self._group_features_by_module(feature_cols)

        if self.config.musical_features:
            missing = set(self.config.musical_features) - set(grouped_features.keys())
            if missing:
                print(f"Warning: modules not found: {missing}")
            grouped_features = {
                m: cols for m, cols in grouped_features.items()
                if m in self.config.musical_features
            }

        module_names = sorted(grouped_features.keys())
        module_index = {m: i for i, m in enumerate(module_names)}
        feature_to_module: Dict[str, str] = {
            col: module
            for module, cols in grouped_features.items()
            for col in cols
        }
        feat_mod_ids = np.array(
            [module_index[feature_to_module[c]] for c in feature_cols
             if c in feature_to_module]
        )
        # Only consider features that belong to a (possibly filtered) module
        valid_feat_mask = np.array([c in feature_to_module for c in feature_cols])

        # 3. Attribute each PC to its dominant module via squared loadings
        #    loadings: (n_global_pcs, n_features)
        loadings_sq = self.global_pca.components_[:, valid_feat_mask] ** 2

        n_modules = len(module_names)
        module_scores = np.zeros((n_global, n_modules))
        for mod_i in range(n_modules):
            mask = feat_mod_ids == mod_i
            n_feats_in_mod = mask.sum()
            if n_feats_in_mod == 0:
                continue
            # Mean (not sum) squared loading per feature — size-invariant so
            # large modules (e.g. melody with 11k features) don't dominate
            # small ones (e.g. scale with a few hundred).
            module_scores[:, mod_i] = loadings_sq[:, mask].sum(axis=1) / n_feats_in_mod

        dominant_module_ids = module_scores.argmax(axis=1)   # (n_pcs,)
        pc_to_module = [module_names[i] for i in dominant_module_ids]
        self.pc_to_module: Dict[int, str] = {i: m for i, m in enumerate(pc_to_module)}
        self.feature_cols_: List[str] = feature_cols

        # 4. Build per-module DataFrames from attributed PCs
        module_pc_indices: Dict[str, List[int]] = defaultdict(list)
        for pc_idx, module in enumerate(pc_to_module):
            module_pc_indices[module].append(pc_idx)

        for module in sorted(module_pc_indices.keys()):
            pc_indices = module_pc_indices[module]
            pc_labels  = [f"PC{rank+1}" for rank in range(len(pc_indices))]
            columns    = pd.MultiIndex.from_product([[module], pc_labels])
            df_module  = pd.DataFrame(
                X_pca[:, pc_indices], index=df.index, columns=columns
            )
            self.module_dataframes[module] = df_module

            mod_explained = float(
                self.global_pca.explained_variance_ratio_[pc_indices].sum()
            )
            self.summary_rows.append({
                "module":                  module,
                "orig_dim":                len(grouped_features[module]),
                "reduced_dim":             len(pc_indices),
                "explained_var_ratio_sum": mod_explained,
                "note":                    "attributed from global PCA",
            })
            print(f"  {module:12s}: {len(grouped_features[module]):3d} orig feats → "
                  f"{len(pc_indices):3d} global PCs attributed "
                  f"({mod_explained:6.1%} variance)")

        # 5. Assemble outputs
        df_stage1 = self._get_stage1_dataframe()
        pc_names  = [f"global__PC{i+1}" for i in range(n_global)]
        df_final  = pd.DataFrame(X_pca, index=df.index, columns=pc_names)

        return df_stage1, df_final

    # ─────────────────────────────────────────────────────────────────────────
    # Modular modes
    # ─────────────────────────────────────────────────────────────────────────

    def _stage1_per_module_pca(self, df: pd.DataFrame) -> None:
        """Stage 1: Apply per-module PCA with optional caps."""
        feature_cols = df.columns
        grouped_features = self._group_features_by_module(feature_cols)

        if self.config.musical_features:
            missing = set(self.config.musical_features) - set(grouped_features.keys())
            if missing:
                print(f"Warning: modules not found: {missing}")
            grouped_features = {
                m: cols for m, cols in grouped_features.items()
                if m in self.config.musical_features
            }

        caps_to_use = self.config.get_effective_caps()

        for module, cols in grouped_features.items():
            self._fit_module_pca(df, module, cols, caps_to_use)

    def _fit_module_pca(
        self,
        df: pd.DataFrame,
        module: str,
        cols: List[str],
        caps: Dict[str, int],
    ) -> None:
        """Fit PCA for a single module."""
        if len(cols) < 2:
            if self.config.keep_singletons and cols:
                self._handle_singleton(df, module, cols)
            return

        X_module = df[cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_module)

        pca = PCA(n_components=self.config.var_ratio_per_module, svd_solver="full")
        X_pca_tmp = pca.fit_transform(X_scaled)
        n_components = X_pca_tmp.shape[1]

        cap = caps.get(module)
        if cap and n_components > cap:
            n_components = cap
            pca = PCA(n_components=n_components, svd_solver="full")
            X_pca = pca.fit_transform(X_scaled)
        else:
            X_pca = X_pca_tmp

        self.scalers[module] = scaler
        self.pcas[module] = pca

        pc_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        columns  = pd.MultiIndex.from_product([[module], pc_names])
        df_module = pd.DataFrame(X_pca, index=df.index, columns=columns)
        self.module_dataframes[module] = df_module

        explained_var = float(np.sum(pca.explained_variance_ratio_))
        self.summary_rows.append({
            "module":                  module,
            "orig_dim":                X_scaled.shape[1],
            "reduced_dim":             X_pca.shape[1],
            "explained_var_ratio_sum": explained_var,
            "note":                    "",
        })
        print(f"  {module:12s}: {X_scaled.shape[1]:3d} → {X_pca.shape[1]:3d} "
              f"({explained_var:6.1%} variance)")

    def _handle_singleton(
        self,
        df: pd.DataFrame,
        module: str,
        cols: List[str],
    ) -> None:
        """Pass through modules with <2 features."""
        df_single = df[cols].copy()
        df_single.columns = pd.MultiIndex.from_product([[module], cols])
        self.module_dataframes[module] = df_single
        self.summary_rows.append({
            "module":                  module,
            "orig_dim":                len(cols),
            "reduced_dim":             len(cols),
            "explained_var_ratio_sum": np.nan,
            "note":                    "pass-through (singleton)",
        })

    def get_feature_importance(
        self,
        module: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Return feature importances derived from global PCA loadings.

        For each PC attributed to `module` (or all PCs if module=None),
        importance of a feature = weighted sum of squared loadings across
        those PCs, where each PC is weighted by its explained variance ratio,
        normalised to sum to 1.

        Parameters
        ----------
        module  : restrict to PCs attributed to this module; None = all PCs.
        top_n   : return only the top-n features by importance.

        Returns
        -------
        DataFrame with columns [feature, module, importance],
        sorted descending by importance.
        """
        if self.global_pca is None or not hasattr(self, "pc_to_module"):
            raise RuntimeError("Pipeline not fitted, or not in global mode.")

        if module is not None:
            pc_indices = [i for i, m in self.pc_to_module.items() if m == module]
            if not pc_indices:
                raise ValueError(f"No PCs attributed to module '{module}'.")
        else:
            pc_indices = list(self.pc_to_module.keys())

        # components_ shape: (n_pcs, n_features)
        # Weight each PC's squared loadings by its explained variance ratio
        # so dominant PCs contribute proportionally more.
        ev_ratios   = self.global_pca.explained_variance_ratio_[pc_indices]
        loadings_sq = self.global_pca.components_[pc_indices] ** 2  # (k, n_feats)
        importance  = (loadings_sq * ev_ratios[:, None]).sum(axis=0)
        importance /= importance.sum()

        feature_module = {
            col: mod
            for mod, cols in self._group_features_by_module(self.feature_cols_).items()
            for col in cols
        }
        df_imp = pd.DataFrame({
            "feature":    self.feature_cols_,
            "module":     [feature_module.get(c, "unknown") for c in self.feature_cols_],
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return df_imp.head(top_n) if top_n else df_imp

    # ─────────────────────────────────────────────────────────────────────────
    # Shared helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_stage1_dataframe(self, flatten: bool = False) -> pd.DataFrame:
        """Concatenate all per-module dataframes."""
        ordered_modules = list(self.module_dataframes.keys())
        df = pd.concat([self.module_dataframes[m] for m in ordered_modules], axis=1)

        if flatten:
            df.columns = [
                f"{mod}__{pc}" if isinstance(mod, str) else f"{mod[0]}__{mod[1]}"
                for (mod, pc) in df.columns
            ]
        return df

    def _group_features_by_module(self, cols) -> Dict[str, List[str]]:
        """Group features by module using regex patterns."""
        grouped = defaultdict(list)
        for col in cols:
            matched = False
            for pattern, module in module_regex_map.items():
                if re.fullmatch(pattern, col):
                    grouped[module].append(col)
                    matched = True
                    break
            if not matched:
                grouped["unknown"].append(col)
        return grouped