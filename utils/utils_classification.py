import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon

def bootstrap_ci(scores, n_boot=1000, ci=95, random_state=42):
    rng = np.random.default_rng(random_state)
    scores = np.asarray(scores)
    scores_clean = scores[~np.isnan(scores)]
    if len(scores_clean) == 0:
        return (np.nan, np.nan)
    boot = [rng.choice(scores_clean, size=len(scores_clean),
                       replace=True).mean()
            for _ in range(n_boot)]
    lo = np.percentile(boot, (100 - ci) / 2)
    hi = np.percentile(boot, 100 - (100 - ci) / 2)
    return (round(lo, 4), round(hi, 4))


def skill_score_array(f1_model, f1_dummy):
    """Per-split skill score: (model - dummy) / (1 - dummy)."""
    eps   = 1e-6
    denom = 1 - np.asarray(f1_dummy)
    valid = denom >= eps
    scores = np.where(
        valid,
        (np.asarray(f1_model) - np.asarray(f1_dummy)) / np.where(valid, denom, 1.0),
        np.nan
    )
    return scores


def imbalance_ratio(y):
    counts = pd.Series(y).value_counts()
    return round(counts.max() / counts.min(), 3)


def entropy_balance(y):
    counts = pd.Series(y).value_counts(normalize=True)
    n = len(counts)
    if n <= 1:
        return 0.0
    ent = -np.sum(counts * np.log(counts + 1e-9))
    return round(ent / np.log(n), 3)


def _make_grouped_splits(X, y, groups, n_splits, test_size=0.20, random_state=42):
    gss = GroupShuffleSplit(
        n_splits=n_splits * 5,  # more attempts helps when labels are imbalanced
        test_size=test_size,
        random_state=random_state
    )
    all_labels = np.unique(y)
    n_classes  = len(all_labels)
    valid = []

    for tr, te in gss.split(X, y, groups=groups):
        # Train must contain all classes
        if len(np.unique(y[tr])) < n_classes:
            continue
        # Test must contain at least 2 classes and at least some minimum size
        if len(np.unique(y[te])) < 2 or len(te) < 10:
            continue

        valid.append((tr, te))
        if len(valid) == n_splits:
            break

    return valid


def _stratified_dummy_f1(y_tr, y_te, rng, labels=None):
    """
    Stratified dummy baseline:
    sample predictions from the empirical label distribution of y_tr.
    
    Parameters
    ----------
    y_tr, y_te : arrays
    rng        : np.random.Generator
    labels     : array-like or None
        Fixed label universe for macro-F1 scoring. If None, uses labels seen in y_tr.
    """
    labs, counts = np.unique(y_tr, return_counts=True)
    probs = counts / counts.sum()

    y_pred = rng.choice(labs, size=len(y_te), replace=True, p=probs)

    if labels is None:
        labels = labs  # default to training labels

    return f1_score(
        y_te, y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

def _permutation_null_f1(y_tr, y_te, rng):
    """
    Permutation null: randomly permute label identities.
    Equivalent to applying a random bijection to labels.
    """
    labels = np.unique(y_tr)
    perm   = labels[rng.permutation(len(labels))]
    pmap   = dict(zip(labels, perm))
    y_perm = np.array([pmap[l] for l in y_te])  # safe because train contains all labels
    return f1_score(y_te, y_perm, labels=labels, average="macro", zero_division=0)



def evaluate_grouped(
    X, y, groups,
    n_splits=10,
    test_size=0.20,
    random_state=42,
    C=1.0,
    max_iter=5000,
    dummy_mode="stratified",  # "stratified" (main baseline) or "permutation" (null)
):
    """
    Aria-grouped evaluation with a consistent macro-F1 definition
    and a clear dummy baseline.

    Returns
    -------
    dict with:
      f1_model, f1_dummy, skill_arr, p_model_vs_dummy, n_splits_used
    """
    splits = _make_grouped_splits(X, y, groups, n_splits, test_size, random_state)
    if not splits:
        return {
            "f1_model": np.array([]),
            "f1_dummy": np.array([]),
            "skill_arr": np.array([]),
            "p_model_vs_dummy": np.nan,
            "n_splits_used": 0
        }

    rng = np.random.default_rng(random_state)

    # ✅ fixed label universe -> comparable macro-F1 across splits
    all_labels = np.unique(y)

    clf = LogisticRegression(
        C=C, solver="saga", max_iter=max_iter,
        class_weight="balanced", random_state=random_state
    )

    f1_model = []
    f1_dummy = []

    for train_idx, test_idx in splits:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # ---- model ----
        clf.fit(X_tr_s, y_tr)
        y_hat = clf.predict(X_te_s)

        f1_m = f1_score(
            y_te, y_hat,
            labels=all_labels,              # ✅ same labels each split
            average="macro",
            zero_division=0
        )
        f1_model.append(f1_m)

        # ---- dummy ----
        if dummy_mode == "stratified":
            f1_d = _stratified_dummy_f1(y_tr, y_te, rng, labels=all_labels)
        elif dummy_mode == "permutation":
            f1_d = _permutation_null_f1(y_tr, y_te, rng, labels=all_labels)
        else:
            raise ValueError("dummy_mode must be 'stratified' or 'permutation'")
        f1_dummy.append(f1_d)

    f1_model = np.array(f1_model, dtype=float)
    f1_dummy = np.array(f1_dummy, dtype=float)

    skill_arr = skill_score_array(f1_model, f1_dummy)

    # paired one-tailed test: model > dummy
    try:
        p_model_vs_dummy = wilcoxon(f1_model, f1_dummy, alternative="greater").pvalue
    except Exception:
        p_model_vs_dummy = np.nan

    return {
        "f1_model": f1_model,
        "f1_dummy": f1_dummy,
        "skill_arr": skill_arr,
        "p_model_vs_dummy": float(p_model_vs_dummy) if not np.isnan(p_model_vs_dummy) else np.nan,
        "n_splits_used": len(splits),
        "dummy_mode": dummy_mode,
    }