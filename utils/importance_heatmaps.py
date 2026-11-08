import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Consistent categorical palette used for row families in the heatmap.
PALETTE_COLORS = [
    "#4E79A7",  # blue
    "#F28E2B",  # orange
    "#59A14F",  # green
    "#E15759",  # red
    "#76B7B2",  # teal
    "#EDC948",  # yellow
    "#B07AA1",  # purple
    "#FF9DA7",  # pink
    "#9C755F",  # brown
    "#BAB0AC",  # gray
]


def plot_module_importance(
    module_store,
    FAMILIES,
    all_schemes=None,
    regex_map=None,
    *,
    mode="family",                 # "family" or "per_scheme"
    save=False,
    suffix_experiment="",
    schemes=None,
    min_mean_importance=0.0,
    aggregate_rows=True,
    aggregate_cols=True,           # only used when mode="family"
    normalize=True,
    top_n_rows=None,
    cmap=None,                     # defaults: YlGnBu (family), YlOrRd (per_scheme)
    annotate=None,                 # defaults: True (family), False (per_scheme)
    filename_suffix=None,          # defaults: "family" or "scheme"
    show=True,
    palette_colors=None,
):
    """Plot a PCA feature-family importance heatmap.

    Parameters
    ----------
    module_store : dict
        {scheme_name: pd.Series/dict of component -> importance}.
    FAMILIES : dict
        {family_name: [scheme_name, ...]} used to order/aggregate columns.
    all_schemes : list or None
        Default scheme universe. If None, derived from FAMILIES.
    regex_map : dict or None
        Optional fallback regex map (e.g., module_regex_map) for naming
        PCA components that don't match the flattened PC pattern.
    mode : {"family", "per_scheme"}
        "family"     -> columns aggregated by FAMILIES (mean per family).
        "per_scheme" -> one column per scheme, grouped/separated by family.
    Other keyword args mirror the original plotting functions.

    Returns
    -------
    (fig, ax, mat_plot) or (None, None, None) if nothing to plot.
    """
    if mode not in ("family", "per_scheme"):
        raise ValueError("mode must be 'family' or 'per_scheme'")

    if all_schemes is None:
        all_schemes = [s for family in FAMILIES.values() for s in family]
    if palette_colors is None:
        palette_colors = PALETTE_COLORS

    # Mode-specific defaults
    if mode == "family":
        cmap = cmap or "YlGnBu"
        annotate = True if annotate is None else annotate
        filename_suffix = filename_suffix or "family"
    else:
        cmap = cmap or "YlOrRd"
        annotate = False if annotate is None else annotate
        filename_suffix = filename_suffix or "scheme"

    # ---- build rows x schemes matrix ----
    mat, valid_schemes = _build_importance_matrix(
        module_store, all_schemes, regex_map,
        schemes=schemes, aggregate_rows=aggregate_rows,
    )
    if mat.empty:
        print("No schemes found in module_store.")
        return None, None, None

    if mode == "family":
        fig, ax, mat_plot = _plot_family(
            mat, FAMILIES, min_mean_importance, aggregate_cols,
            normalize, top_n_rows, cmap, annotate, palette_colors,
        )
    else:
        fig, ax, mat_plot = _plot_per_scheme(
            mat, FAMILIES, min_mean_importance,
            normalize, top_n_rows, cmap, annotate,
        )

    if fig is None:
        return None, None, None

    if save:
        fname = f"module_importance_heatmap_{filename_suffix}_{suffix_experiment}.pdf"
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        print(f"Saved: {fname}")
    if show:
        plt.show()

    return fig, ax, mat_plot


# ─────────────────────────────────────────────────────────────────────────────
# Matrix construction
# ─────────────────────────────────────────────────────────────────────────────

def _component_to_family(feature_name: str, regex_map=None) -> str:
    """Map a PCA component/feature name to a high-level musical family."""
    s = str(feature_name)

    # Typical flattened PCA names: module__PCk, module_PCk, module_gPCk
    if "__" in s:
        return s.split("__", 1)[0]

    if "_" in s:
        left, right = s.split("_", 1)
        if re.match(r"^(g)?PC\d+$", right):
            return left

    # Fallback: regex map from musif features, if available
    if regex_map is not None:
        for pattern, family in regex_map.items():
            if re.fullmatch(pattern, s):
                return family

    return "unknown"


def aggregate_by_family_pca(mat: pd.DataFrame, regex_map=None) -> pd.DataFrame:
    """Aggregate a component-level matrix by PCA feature family (rows)."""
    if mat is None or mat.empty:
        return pd.DataFrame()

    row_families = [
        _component_to_family(idx, regex_map=regex_map)
        for idx in mat.index
    ]
    return mat.groupby(pd.Index(row_families), sort=False).sum()


def _build_importance_matrix(
    module_store, all_schemes, regex_map, schemes=None, aggregate_rows=True
):
    """Build rows x schemes matrix from module_store."""
    valid_schemes = [s for s in (schemes or all_schemes) if s in module_store]
    if not valid_schemes:
        return pd.DataFrame(), []

    mat = pd.DataFrame({s: module_store[s] for s in valid_schemes}).fillna(0)
    mat = mat.loc[mat.sum(axis=1) > 0]

    if aggregate_rows:
        mat = aggregate_by_family_pca(mat, regex_map=regex_map)

    return mat, valid_schemes


# ─────────────────────────────────────────────────────────────────────────────
# Shared styling helpers
# ─────────────────────────────────────────────────────────────────────────────

def _annotation_threshold(n_cells: int, base=0.03) -> float:
    """Increase annotation threshold as matrices get denser to avoid overlap."""
    if n_cells > 120:
        return max(base, 0.10)
    if n_cells > 70:
        return max(base, 0.07)
    if n_cells > 40:
        return max(base, 0.05)
    return base


def _annotate_heatmap(ax, mat_plot, min_annot=0.03):
    for i in range(mat_plot.shape[0]):
        for j in range(mat_plot.shape[1]):
            value = mat_plot.iat[i, j]
            if value >= min_annot:
                txt_color = "white" if value > 0.28 else "black"
                ax.text(
                    j, i, f"{value:.1%}", ha="center", va="center",
                    fontsize=8, color=txt_color, fontweight="bold"
                )


def _style_axis_labels(ax, xlabels, ylabels, xtick_rotation=25):
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=xtick_rotation, ha="right",
                       fontsize=9, fontweight="bold")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=10)


def _scheme_short_label(s):
    return (s
            .replace("LLM-Prop_Small_", "P-Small ")
            .replace("LLM-Prop_Large_", "P-Large ")
            .replace("LLM-Descartes", "Desc ")
            .replace("LLM-AriEmozione_", "Ari ")
            .replace("LLM-Annotator_A_", "AnnA ")
            .replace("LLM-Annotator_B_", "AnnB ")
            .replace("_Phi4", " (P4)")
            .replace("_Gemma", " (Ge)")
            .replace("AriEmozione", "Ari")
            .replace("Annotator_A", "AnnA")
            .replace("Annotator_B", "AnnB"))


# ─────────────────────────────────────────────────────────────────────────────
# Family-level heatmap
# ─────────────────────────────────────────────────────────────────────────────

def _plot_family(
    mat, FAMILIES, min_mean_importance, aggregate_cols,
    normalize, top_n_rows, cmap, annotate, palette_colors,
):
    if aggregate_cols:
        fam_cols = {}
        for family_name, family_schemes in FAMILIES.items():
            present = [s for s in family_schemes if s in mat.columns]
            if present:
                # Mean keeps each family comparable despite different family sizes
                fam_cols[family_name] = mat[present].mean(axis=1)
        mat = pd.DataFrame(fam_cols)

    if mat.empty:
        print("No data left after family aggregation.")
        return None, None, None

    if min_mean_importance > 0:
        mat = mat.loc[mat.mean(axis=1) >= min_mean_importance]

    if top_n_rows is not None and top_n_rows > 0 and len(mat.index) > top_n_rows:
        mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).head(top_n_rows).index]

    if mat.empty:
        print("No modules left after filtering.")
        return None, None, None

    if normalize:
        mat_plot = mat.div(mat.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)
    else:
        mat_plot = mat.copy()

    mat_plot = mat_plot.loc[mat_plot.mean(axis=1).sort_values(ascending=False).index]

    fig_w = max(9, len(mat_plot.columns) * 2.6)
    fig_h = max(5.5, len(mat_plot.index) * 0.75 + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    im = ax.imshow(mat_plot.values, aspect="auto", cmap=cmap)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label(
        "Normalized importance" if normalize else "Importance",
        fontsize=10, fontweight="bold",
    )

    _style_axis_labels(ax, mat_plot.columns, mat_plot.index, xtick_rotation=20)
    ax.tick_params(axis="y", pad=16)

    if annotate:
        min_annot = _annotation_threshold(mat_plot.shape[0] * mat_plot.shape[1], base=0.03)
        _annotate_heatmap(ax, mat_plot, min_annot=min_annot)

    # Row-family color strip on the left
    row_colors = {
        fam: palette_colors[i % len(palette_colors)]
        for i, fam in enumerate(mat_plot.index)
    }
    for i, fam in enumerate(mat_plot.index):
        ax.add_patch(
            mpatches.Rectangle(
                (-0.50, i - 0.5), 0.03, 1.0,
                facecolor=row_colors[fam], edgecolor="none", clip_on=False
            )
        )

    ax.set_xlabel("Annotation families", fontsize=9, fontweight="bold")
    ax.set_ylabel("PCA feature families", fontsize=9, fontweight="bold")
    ax.set_title(
        "PCA feature-family importance by annotation family\n"
        "(rows aggregated by PCA module, columns aggregated by FAMILIES)",
        fontsize=12, fontweight="bold", pad=12,
    )

    return fig, ax, mat_plot


# ─────────────────────────────────────────────────────────────────────────────
# Per-scheme heatmap
# ─────────────────────────────────────────────────────────────────────────────

def _plot_per_scheme(
    mat, FAMILIES, min_mean_importance,
    normalize, top_n_rows, cmap, annotate,
):
    # Keep family-defined order in columns for readability
    ordered_cols = [s for fam in FAMILIES.values() for s in fam if s in mat.columns]
    mat = mat[ordered_cols]

    if min_mean_importance > 0:
        mat = mat.loc[mat.mean(axis=1) >= min_mean_importance]

    if top_n_rows is not None and top_n_rows > 0 and len(mat.index) > top_n_rows:
        mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).head(top_n_rows).index]

    if mat.empty:
        print("No modules left after filtering.")
        return None, None, None

    if normalize:
        mat_plot = mat.div(mat.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)
    else:
        mat_plot = mat.copy()

    mat_plot = mat_plot.loc[mat_plot.mean(axis=1).sort_values(ascending=False).index]

    fig_w = max(14, len(mat_plot.columns) * 0.9)
    fig_h = max(6, len(mat_plot.index) * 0.8 + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    im = ax.imshow(mat_plot.values, aspect="auto", cmap=cmap)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(
        "Normalized importance" if normalize else "Importance",
        fontsize=10, fontweight="bold",
    )

    short_cols = [_scheme_short_label(s) for s in mat_plot.columns]
    _style_axis_labels(ax, short_cols, mat_plot.index, xtick_rotation=30)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=12)

    # Family separators and labels
    family_counts = [len([s for s in fam if s in mat_plot.columns]) for fam in FAMILIES.values()]
    separators = np.cumsum(family_counts)[:-1] - 0.5
    for x in separators:
        ax.axvline(x, color="white", linewidth=2.2)

    starts = np.cumsum([0] + family_counts[:-1])
    centers = starts + (np.array(family_counts) / 2) - 0.5
    for center, fname in zip(centers, FAMILIES.keys()):
        if center >= -0.5:
            ax.text(
                center, -0.55, fname,
                ha="center", va="top", fontsize=9, fontweight="bold",
                transform=ax.get_xaxis_transform()
            )

    if annotate:
        min_annot = _annotation_threshold(mat_plot.shape[0] * mat_plot.shape[1], base=0.07)
        _annotate_heatmap(ax, mat_plot, min_annot=min_annot)

    ax.set_xlabel("Annotation schemes", fontsize=11, fontweight="bold", labelpad=18)
    ax.set_ylabel("PCA feature families", fontsize=11, fontweight="bold")
    ax.set_title(
        "PCA feature-family importance by individual annotation scheme\n"
        "(rows aggregated by PCA module)",
        fontsize=12, fontweight="bold", pad=14,
    )

    return fig, ax, mat_plot