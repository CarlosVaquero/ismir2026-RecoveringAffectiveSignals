# Recovering Affective Signals in Baroque Opera

Code and data accompanying our paper at **ISMIR 2026**.

> **Recovering Affective Signals in Baroque Opera: A Comparative Evaluation of Human and LLM Annotation**
> Carlos Vaquero Patricio, Álvaro Torrente
> Proceedings of the 27th International Society for Music Information Retrieval Conference (ISMIR), Abu Dhabi, UAE, 2026.

📄 Paper: https://ismir2026program.ismir.net/poster_206.html
💻 Repository: https://github.com/CarlosVaquero/ismir2026-RecoveringAffectiveSignals

---

## Overview

This repository contains the code and data used to produce the experiments and figures
in the paper. The study compares human and LLM-produced annotation schemes for affective
content in Baroque opera arias, evaluating how well each scheme's emotion categories are
recovered from `musif`-derived musical features. It covers:

- **Feature grouping** — mapping `musif` features to high-level musical modules
  (harmony, melody, rhythm, dynamics, texture, core).
- **LLM annotation** — tagging arias with emotion labels via LLMs
  (`utils_llm_tagging.py`, configured in `llm_config.py`).
- **Dimensionality reduction** — PCA over the feature set.
- **Classification** — aria-grouped, macro-F1 evaluation with skill scores against
  stratified and permutation baselines.
- **Semantic validity** — label-centroid distances and pairwise Adjusted Rand Index (ARI)
  across annotation schemes.
- **Visualization** — PCA feature-family importance heatmaps.

## Repository structure

*(Adjust to match your actual layout — TODO)*

```
.
├── Experiments_notebook_rec_af_signals.ipynb   # Main notebook — reproduces all paper results
├── constants.py                    # Thresholds, IDs, column exclusion lists, feature subsets
├── group_musif_features.py         # Regex map: musif feature name -> musical module
├── feature_reduction.py            # PCA reduction pipeline wrapper
├── utils_classification.py         # Grouped evaluation, skill scores, baselines, bootstrap CIs
├── semantic_validity_metrics.py    # Centroid distances + pairwise ARI
├── module_importance_heatmaps.py   # PCA feature-family importance heatmaps
├── utils_llm_tagging.py            # LLM tagging/annotation functions
├── llm_config.py                   # LLM configuration (models, prompts, API settings)
├── utils/
│   ├── modeling_functions.py       # PCAConfig, PCAReductionPipeline, ...
│   ├── preprocessing_functions.py
│   ├── evaluation.py
│   └── utils_outliers.py
├── Arias_texts_and_annotations/    # Aria texts and LLM annotations
└── supplementary_material.pdf      # Extended experiments and results (TODO: confirm filename)
```

## Installation

```bash
git clone https://github.com/CarlosVaquero/ismir2026-RecoveringAffectiveSignals.git
cd ismir2026-RecoveringAffectiveSignals

conda env create -f environment.yml
conda activate recovering_affective_signals
```

## Usage

The **experiments notebook** (`Experiments_notebook_rec_af_signals.ipynb`) replicates the
results reported in the paper end to end: it downloads the data, builds the feature
matrices, runs the classification and semantic-validity evaluations, and produces the
figures. Run its cells in order to reproduce the paper's results. The helper modules
below are imported by the notebook and can also be called directly.

**Classification evaluation**

```python
from utils_classification import evaluate_grouped

result = evaluate_grouped(X, y, groups, n_splits=10, dummy_mode="stratified")
print(result["f1_model"].mean(), result["skill_arr"].mean())
```

**Semantic validity metrics**

```python
from semantic_validity_metrics import (
    compute_centroid_distances, print_centroid_report,
    compute_pairwise_ari, print_ari_report,
)

result = compute_centroid_distances(X, y, scheme_name="PassionA")
print_centroid_report(result)

ari_df = compute_pairwise_ari(schemes_dict)
print_ari_report(ari_df)
```

**Importance heatmaps**

```python
from module_importance_heatmaps import plot_module_importance

all_schemes = [s for family in FAMILIES.values() for s in family]

# Family-aggregated view
plot_module_importance(
    module_store, FAMILIES, all_schemes,
    regex_map=module_regex_map, mode="family",
    normalize=True, annotate=True,
)

# Per-scheme view
plot_module_importance(
    module_store, FAMILIES, all_schemes,
    regex_map=module_regex_map, mode="per_scheme",
    normalize=True, annotate=True,
)
```

Pass `save=True, suffix_experiment="..."` to write figures to PDF.

## Supplementary material

The supplementary material PDF is included, covering in full the experiments and results
presented and discussed in the paper. **Supplementary_material_A_Comparative_Evaluation_of_Human_and_LLM_Annotation.pdf**

## Data

The dataset is downloaded automatically from a shared drive within the experiments
notebook — no manual download step is required; run the notebook's data-loading cell
and it will fetch the files.

The aria texts and the LLM annotations are provided in **`Arias_texts_and_annotations`**.


## Citation

If you use this code or data, please cite:

```bibtex
@inproceedings{vaquero2026recovering,
  title     = {Recovering Affective Signals in Baroque Opera: A Comparative Evaluation of Human and LLM Annotation},
  author    = {Vaquero Patricio, Carlos and Torrente, {\'A}lvaro},
  booktitle = {Proceedings of the 27th International Society for Music Information Retrieval Conference (ISMIR)},
  address   = {Abu Dhabi, UAE},
  year      = {2026},
}
```
