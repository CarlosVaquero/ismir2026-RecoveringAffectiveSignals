# Custom utilities

import sys
sys.path.append("..")
from utils.unused_utils.preprocessing_functions import *

from scipy.stats import entropy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def tune_hybrid_weights_KL(M, OC, L, step=0.1, tail_frac=0.2, bins=40):
    best_kl = -np.inf
    best_weights = (1/3, 1/3, 1/3)

    vals = np.arange(0, 1 + 1e-9, step)

    for a in vals:
        for b in vals:
            c = 1 - a - b
            if c < 0:
                continue

            H = a*M + b*OC + c*L
            H_sorted = np.sort(H)
            k = max(1, int(tail_frac * len(H)))
            tail = H_sorted[-k:]
            bulk = H_sorted[:-k]

            # Histogram PDFs
            p_tail, edges = np.histogram(tail, bins=bins, density=True)
            p_bulk, _     = np.histogram(bulk, bins=edges, density=True)

            # Avoid zero probabilities
            p_tail += 1e-10
            p_bulk += 1e-10

            kl = entropy(p_tail, p_bulk)
            if kl > best_kl:
                best_kl = kl
                best_weights = (a, b, c)

    a, b, c = best_weights
    hybrid_score = a*M + b*OC + c*L
    return a, b, c, hybrid_score, best_kl

def plot_scores(X_2d, scores, title, outlier_mask, aria_name):
    plt.figure(figsize=(6, 5))

    # Normal points
    plt.scatter(
        X_2d[~outlier_mask, 0],
        X_2d[~outlier_mask, 1],
        c=scores[~outlier_mask],
        cmap="viridis",
        s=60,
        alpha=0.6
    )

    # Outliers
    plt.scatter(
        X_2d[outlier_mask, 0],
        X_2d[outlier_mask, 1],
        c=scores[outlier_mask],
        cmap="viridis",
        s=120,
        edgecolor='red',
        linewidths=2,
        marker='o'
    )

    plt.title(f"{title} — {aria_name}")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.colorbar(label="Hybrid Score")
    plt.grid(alpha=0.2)
    plt.show()

def select_percentile_KL(
    hybrid_score,
    candidate_percentiles=(80, 85, 90, 95),
    tail_frac_min=2,
    bins=30
):
    """
    Select outlier percentile by maximizing KL divergence between
    tail and bulk of hybrid anomaly scores.
    """

    best_p = None
    best_kl = -np.inf

    H = np.asarray(hybrid_score)

    for p in candidate_percentiles:
        th = np.percentile(H, p)

        tail = H[H >= th]
        bulk = H[H < th]

        # safety for very small samples
        if len(tail) < tail_frac_min or len(bulk) < tail_frac_min:
            continue

        # histograms on shared bins
        p_tail, edges = np.histogram(tail, bins=bins, density=True)
        p_bulk, _     = np.histogram(bulk, bins=edges, density=True)

        # numerical stability
        p_tail += 1e-10
        p_bulk += 1e-10

        kl = entropy(p_tail, p_bulk)

        if kl > best_kl:
            best_kl = kl
            best_p = p

    # fallback (important!)
    if best_p is None:
        best_p = 90
        best_kl = np.nan

    return best_p, best_kl

def bootstrap_ci(
    values,
    n_bootstrap=10000,
    ci=0.95,
    random_state=42
):
    """
    Percentile bootstrap confidence interval for the mean.
    """
    rng = np.random.default_rng(random_state)
    boot_means = rng.choice(
        values,
        size=(n_bootstrap, len(values)),
        replace=True
    ).mean(axis=1)

    alpha = (1 - ci) / 2
    lower = np.quantile(boot_means, alpha)
    upper = np.quantile(boot_means, 1 - alpha)

    return lower, upper


def plot_umap(df_plot):
    # plt.figure(figsize=(7, 5), dpi=125)

    plt.figure(figsize=(900/100, 700/100), dpi=120)

    # Non-outliers
    mask = ~df_plot["IsOutlier"]
    plt.scatter(
        df_plot.loc[mask, "UMAP1"],
        df_plot.loc[mask, "UMAP2"],
        s=18,
        c=df_plot.loc[mask, "AriaName"].astype("category").cat.codes,
        cmap="tab10",
        alpha=0.55
    )

    # Outliers
    mask_o = df_plot["IsOutlier"]
    plt.scatter(
        df_plot.loc[mask_o, "UMAP1"],
        df_plot.loc[mask_o, "UMAP2"],
        s=60,
        facecolor='none',
        edgecolor='red',
        linewidth=1.3,
        label="Outliers"
    )

    # Labels
    plt.xlabel("UMAP–1", fontsize=12)
    plt.ylabel("UMAP–2", fontsize=12)
    plt.title("Hybrid Anomaly Map – All Arias (Outliers Circled)", fontsize=14)

    # Legend entries (arias)
    unique_arias = df_plot["AriaName"].unique()
    cmap = plt.cm.tab10
    for i, aria in enumerate(unique_arias):
        plt.scatter([], [], c=[cmap(i/10)], label=aria, s=35)

    # Outlier marker
    plt.scatter([], [], facecolor='none', edgecolor='red', s=60, label="Outlier")

    plt.legend(fontsize=8, frameon=False, loc='upper right', bbox_to_anchor=(1.35, 1.02))
    plt.tight_layout()