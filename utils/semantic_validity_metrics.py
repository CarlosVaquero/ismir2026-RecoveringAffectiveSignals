"""
semantic_validity_metrics.py
============================
Two complementary validity metrics for annotation scheme comparison:

1. Label centroid distances
   For each scheme, computes pairwise cosine distances between
   per-label embedding centroids. High distance = categories are
   semantically distinct in text space. Complements silhouette score
   by showing WHICH specific label pairs are close or far apart.

2. Adjusted Rand Index (ARI)
   For each pair of schemes sharing the same arias, computes how much
   they agree on which arias belong together — regardless of label names.
   ARI = 1.0 → perfect agreement on grouping structure
   ARI = 0.0 → agreement no better than chance
   ARI < 0   → less agreement than chance

Usage
-----
    from semantic_validity_metrics import (
        compute_centroid_distances,
        compute_pairwise_ari,
        print_centroid_report,
        print_ari_report,
    )

    # Centroid distances — one scheme at a time
    result = compute_centroid_distances(X, y, scheme_name="PassionA")
    print_centroid_report(result)

    # ARI — across all scheme pairs
    # schemes_dict = {scheme_name: (X, y, aria_ids)}
    ari_df = compute_pairwise_ari(schemes_dict)
    print_ari_report(ari_df)
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import adjusted_rand_score


# ─────────────────────────────────────────────────────────────────────────────
# 1. LABEL CENTROID DISTANCES
# ─────────────────────────────────────────────────────────────────────────────

def compute_centroid_distances(X, y, scheme_name=""):
    """
    Compute pairwise cosine distances between per-label embedding centroids.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame, shape (n_samples, n_features)
        Embedding matrix.
    y : pd.Series or np.ndarray, shape (n_samples,)
        Label vector aligned with X.
    scheme_name : str
        Name of the scheme (for reporting).

    Returns
    -------
    dict with keys:
        scheme_name     : str
        labels          : list of label names
        centroids       : np.ndarray (n_labels, n_features)
        distance_matrix : np.ndarray (n_labels, n_labels) — cosine distances
        distance_df     : pd.DataFrame — labelled distance matrix
        mean_distance   : float — mean pairwise distance (off-diagonal)
        min_distance    : float — minimum pairwise distance (closest pair)
        max_distance    : float — maximum pairwise distance (furthest pair)
        closest_pair    : tuple(str, str) — label pair with smallest distance
        furthest_pair   : tuple(str, str) — label pair with largest distance
        per_pair        : list of (label_a, label_b, distance) sorted ascending
    """
    X = np.array(X)
    y = np.array(y)
    labels = sorted(np.unique(y))
    n = len(labels)

    centroids = np.array([X[y == lbl].mean(axis=0) for lbl in labels])
    dist_matrix = cosine_distances(centroids)

    # Off-diagonal indices
    triu_idx = np.triu_indices(n, k=1)
    off_diag = dist_matrix[triu_idx]

    per_pair = sorted(
        [(labels[i], labels[j], dist_matrix[i, j])
         for i, j in zip(*triu_idx)],
        key=lambda x: x[2]
    )

    dist_df = pd.DataFrame(dist_matrix, index=labels, columns=labels)

    return {
        "scheme_name":     scheme_name,
        "labels":          labels,
        "centroids":       centroids,
        "distance_matrix": dist_matrix,
        "distance_df":     dist_df,
        "mean_distance":   float(off_diag.mean()),
        "min_distance":    float(off_diag.min()),
        "max_distance":    float(off_diag.max()),
        "closest_pair":    (per_pair[0][0],  per_pair[0][1]),
        "furthest_pair":   (per_pair[-1][0], per_pair[-1][1]),
        "per_pair":        per_pair,
    }


def print_centroid_report(result, decimals=3):
    """Pretty-print the centroid distance result for one scheme."""
    name  = result["scheme_name"]
    pairs = result["per_pair"]
    width = max(len(f"{a} ↔ {b}") for a, b, _ in pairs) + 2

    print(f"\n{'=' * 55}")
    print(f"  CENTROID DISTANCES  |  {name}")
    print(f"{'=' * 55}")
    print(f"  Mean pairwise distance : {result['mean_distance']:.{decimals}f}")
    print(f"  Closest pair           : {result['closest_pair'][0]} ↔ "
          f"{result['closest_pair'][1]}  "
          f"(d={result['min_distance']:.{decimals}f})")
    print(f"  Furthest pair          : {result['furthest_pair'][0]} ↔ "
          f"{result['furthest_pair'][1]}  "
          f"(d={result['max_distance']:.{decimals}f})")
    print(f"\n  All pairs (ascending):")
    for a, b, d in pairs:
        label = f"  {a} ↔ {b}"
        bar_len = int(d * 40)
        bar = "█" * bar_len
        print(f"  {label:<{width}}  {d:.{decimals}f}  {bar}")
    print(f"\n  Full distance matrix:")
    print(result["distance_df"].round(decimals).to_string())


def summarise_centroid_distances(results_list):
    """
    Combine centroid distance results for multiple schemes into a summary df.

    Parameters
    ----------
    results_list : list of dicts returned by compute_centroid_distances

    Returns
    -------
    pd.DataFrame with one row per scheme
    """
    rows = []
    for r in results_list:
        rows.append({
            "Scheme":         r["scheme_name"],
            "Mean_dist":      round(r["mean_distance"], 4),
            "Min_dist":       round(r["min_distance"],  4),
            "Max_dist":       round(r["max_distance"],  4),
            "Closest_pair":   f"{r['closest_pair'][0]} ↔ {r['closest_pair'][1]}",
            "Furthest_pair":  f"{r['furthest_pair'][0]} ↔ {r['furthest_pair'][1]}",
        })
    return pd.DataFrame(rows).sort_values("Mean_dist", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ADJUSTED RAND INDEX
# ─────────────────────────────────────────────────────────────────────────────

def compute_pairwise_ari(schemes_dict, condition="without_outliers"):
    """
    Compute ARI between all pairs of annotation schemes.

    Because different schemes cover different subsets of arias, we compute
    ARI only on the intersection of arias shared by each pair.

    Parameters
    ----------
    schemes_dict : dict
        Keys   : scheme name (str)
        Values : dict with keys:
                   'y'        → pd.Series of labels, indexed by aria_id
                   'aria_ids' → array-like of aria identifiers (same order as y)

        Example:
            {
              "PassionA": {
                  "y":        pd.Series([...], index=aria_ids),
                  "aria_ids": aria_ids_array
              },
              ...
            }

    condition : str
        Label for the condition (used in output column). E.g.
        "without_outliers" or "with_outliers".

    Returns
    -------
    pd.DataFrame with columns:
        Scheme_A, Scheme_B, N_shared_arias, ARI, Interpretation
    """
    scheme_names = list(schemes_dict.keys())
    rows = []

    for name_a, name_b in combinations(scheme_names, 2):
        data_a = schemes_dict[name_a]
        data_b = schemes_dict[name_b]

        # Build Series indexed by aria_id
        y_a = pd.Series(
            data_a["y"].values,
            index=data_a["aria_ids"]
        )
        y_b = pd.Series(
            data_b["y"].values,
            index=data_b["aria_ids"]
        )

        # Intersection of aria IDs
        shared = y_a.index.intersection(y_b.index)
        n_shared = len(shared)

        if n_shared < 10:
            ari = np.nan
            interp = "Too few shared arias"
        else:
            ari = adjusted_rand_score(y_a[shared], y_b[shared])
            interp = _interpret_ari(ari)

        rows.append({
            "Scheme_A":       name_a,
            "Scheme_B":       name_b,
            "N_shared_arias": n_shared,
            "ARI":            round(ari, 4) if not np.isnan(ari) else np.nan,
            "Interpretation": interp,
        })

    df = pd.DataFrame(rows).sort_values("ARI", ascending=False)
    return df


def _interpret_ari(ari):
    """Map ARI value to a plain-language interpretation."""
    if ari >= 0.70:
        return "Strong agreement"
    elif ari >= 0.40:
        return "Moderate agreement"
    elif ari >= 0.10:
        return "Weak agreement"
    elif ari >= -0.05:
        return "Chance-level agreement"
    else:
        return "Less than chance"


def print_ari_report(ari_df):
    """Pretty-print the pairwise ARI table."""
    print(f"\n{'=' * 75}")
    print(f"  PAIRWISE ADJUSTED RAND INDEX")
    print(f"{'=' * 75}")
    print(f"  ARI = 1.0 → perfect agreement on grouping structure")
    print(f"  ARI = 0.0 → chance-level agreement")
    print(f"  ARI < 0   → worse than chance")
    print()
    print(ari_df.to_string(index=False))


def ari_heatmap_data(ari_df, schemes=None):
    """
    Pivot ARI results into a symmetric matrix suitable for heatmap plotting.

    Parameters
    ----------
    ari_df  : pd.DataFrame from compute_pairwise_ari
    schemes : list of scheme names to include (None = all)

    Returns
    -------
    pd.DataFrame — symmetric ARI matrix with 1.0 on diagonal
    """
    if schemes is None:
        schemes = sorted(
            set(ari_df["Scheme_A"].tolist() + ari_df["Scheme_B"].tolist())
        )

    mat = pd.DataFrame(np.nan, index=schemes, columns=schemes)
    np.fill_diagonal(mat.values, 1.0)

    for _, row in ari_df.iterrows():
        a, b = row["Scheme_A"], row["Scheme_B"]
        if a in schemes and b in schemes:
            mat.loc[a, b] = row["ARI"]
            mat.loc[b, a] = row["ARI"]

    return mat


# ─────────────────────────────────────────────────────────────────────────────
# 3. COMBINED VALIDITY SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def combined_validity_summary(
    centroid_results,
    silhouette_dict,
    skill_dict,
):
    """
    Merge centroid distances, silhouette scores, and skill scores
    into a single summary table for the paper.

    Parameters
    ----------
    centroid_results : list of dicts from compute_centroid_distances
    silhouette_dict  : {scheme_name: silhouette_score}
    skill_dict       : {scheme_name: skill_score}

    Returns
    -------
    pd.DataFrame sorted by Skill_score descending
    """
    rows = []
    for r in centroid_results:
        name = r["scheme_name"]
        rows.append({
            "Scheme":          name,
            "Skill_score":     round(skill_dict.get(name, np.nan), 3),
            "Silhouette":      round(silhouette_dict.get(name, np.nan), 3),
            "Mean_centroid_dist": round(r["mean_distance"], 3),
            "Min_centroid_dist":  round(r["min_distance"],  3),
            "Closest_pair":    f"{r['closest_pair'][0]} ↔ {r['closest_pair'][1]}",
        })

    df = pd.DataFrame(rows).sort_values("Skill_score", ascending=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE INTEGRATION (shows how to call from your main script)
# ─────────────────────────────────────────────────────────────────────────────
"""
Example — add this to your main evaluation script:

from semantic_validity_metrics import (
    compute_centroid_distances,
    summarise_centroid_distances,
    compute_pairwise_ari,
    print_centroid_report,
    print_ari_report,
    combined_validity_summary,
    ari_heatmap_data,
)

# --- CENTROID DISTANCES ---
# Run for each scheme on the without-outliers data
centroid_results = []
for key, (X_all, y_all, X_clean, y_clean, aria_ids_all, aria_ids_clean) in data.items():
    result = compute_centroid_distances(X_clean, y_clean, scheme_name=key)
    print_centroid_report(result)
    centroid_results.append(result)

summary_df = summarise_centroid_distances(centroid_results)
print(summary_df.to_string(index=False))

# --- ARI ---
# Build schemes_dict — y must be a Series indexed by aria_id
schemes_dict = {}
for key, (X_all, y_all, X_clean, y_clean, aria_ids_all, aria_ids_clean) in data.items():
    y_series = pd.Series(y_clean.values, name=key)
    schemes_dict[key] = {
        "y":        y_series,
        "aria_ids": aria_ids_clean,
    }

ari_df = compute_pairwise_ari(schemes_dict, condition="without_outliers")
print_ari_report(ari_df)

# Heatmap matrix for plotting
heatmap = ari_heatmap_data(ari_df)
print(heatmap.round(3).to_string())
"""
