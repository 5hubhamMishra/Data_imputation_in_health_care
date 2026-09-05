# Progress Report

## Project Title

Healthcare Data Imputation Using Machine Learning with Genetic Algorithm-Based Feature Selection

## Objective

Investigate how missing healthcare data affects prediction performance, compare
imputation methods, and determine whether GA-based feature selection improves
downstream Random Forest prediction.

## Research Questions

RQ1-RQ4 as stated in README.md.

## Work Completed

- **IMPLEMENTED, EXECUTED, VALIDATED**: Repository scaffold (directory
  structure, README, requirements.txt, .gitignore).
- **IMPLEMENTED, EXECUTED, VALIDATED**: Dataset discovery — 5 candidates
  compared against 12 criteria (`results/dataset_candidate_comparison.csv`,
  `docs/dataset_selection.md`).
- **IMPLEMENTED, EXECUTED, VALIDATED**: Primary dataset acquired (Heart
  Failure Clinical Records, UCI, CC BY 4.0), integrity-checked, and frozen.
- **IMPLEMENTED, EXECUTED, VALIDATED**: Data loading (`src/data_loader.py`),
  config (`src/config.py`), and leakage-safe train/test split
  (`src/preprocessing.py`), covered by 3 passing tests
  (`tests/test_data_loader.py`).
- **IMPLEMENTED, EXECUTED, VALIDATED**: EDA figures (class distribution,
  feature distributions, correlation matrix) — `results/figures/`.
- **IMPLEMENTED, EXECUTED, VALIDATED**: E0 complete-data Random Forest
  baseline — `results/metrics/e0_complete_rf_baseline.json`.
- **IMPLEMENTED, EXECUTED, VALIDATED**: MCAR missingness framework
  (`src/missingness.py`) at 10/20/30% on the training split only, ground
  truth preserved for intentionally hidden cells, 4 passing tests
  (`tests/test_missingness.py`). Scoped to the 7 continuous features
  (excludes the 5 binary clinical flags — see `src/config.py`).
- **IMPLEMENTED, EXECUTED, VALIDATED**: Mean/median imputation
  (`src/imputation.py`), scored by MAE/RMSE per feature/mechanism/level
  against preserved ground truth only — `results/metrics/e1_mean_median_imputation.csv`.

## Current Dataset and Characteristics

Heart Failure Clinical Records (UCI, id 519). 299 rows, 12 features,
target `DEATH_EVENT` (binary). Fully complete: 0 missing values, 0
duplicates. Class balance 203 (0) / 96 (1), roughly 68%/32%. SHA-256
checksum and full provenance recorded in
`data/raw/heart_failure_metadata.json`.

One column, `time` (days of follow-up), is flagged as leakage-relevant: in
survival-style clinical data, short follow-up correlates with the death
event almost by construction. It is kept in the dataset (removing it would
be an undocumented methodological choice) but its outsized influence on RF
performance must be called out explicitly in the results discussion, not
presented as a genuine predictive discovery.

## Methodology

Standard scikit-learn Random Forest, 80/20 stratified train/test split,
seed 42, 5-fold stratified CV on the training set only. No missingness or
imputation yet — this run establishes the complete-data reference point
required before Phase 8 (missingness framework).

## Experiments Completed

- E0: Complete-data RF baseline (1 run, seed 42).
- Missingness framework: MCAR at 10/20/30% on the training split (seed 42).
- E1: Mean and median imputation on each MCAR level, scored on hidden cells.

## Actual Current Results

E0 (seed 42, held-out test, n=60): Accuracy 0.817, Precision 0.786, Recall
0.579, F1 0.667, ROC-AUC 0.883, CV F1 (train) 0.768 ± 0.082. Full JSON in
`results/metrics/e0_complete_rf_baseline.json`. These numbers are a single
seed and will be extended with the multi-seed manifest in a later phase —
not yet a final claim about model quality.

MCAR realized missingness (training split, n=239): 10.04%, 20.08%, 30.13%
per continuous feature (identical across features by construction — same
row count masked per column). Requested-vs-actual is close throughout; see
`results/tables/missingness_summary.csv`.

E1 mean/median imputation MAE/RMSE (per-feature detail in
`results/metrics/e1_mean_median_imputation.csv`; scale varies hugely by
feature, e.g. platelets is O(10^4) vs serum_creatinine is O(1), so a single
cross-feature average is not analytically meaningful — per-feature values
are the reportable artifact). Mean and median perform similarly at every
level, as expected for this dataset. No RF-after-imputation run yet — that
is Phase 13, not in scope this cycle.

## GA Status

Not started. Scheduled for a later cycle (master prompt phase 14+),
after the missingness/imputation framework exists.

## Important Tables/Figures

- `results/dataset_candidate_comparison.csv`
- `results/figures/class_distribution.png`
- `results/figures/feature_distributions.png`
- `results/figures/correlation_matrix.png`
- `results/figures/missing_per_feature_mcar20.png`
- `results/figures/missingness_heatmap_mcar20.png`
- `results/metrics/e0_complete_rf_baseline.json`
- `results/tables/missingness_summary.csv`
- `results/metrics/e1_mean_median_imputation.csv`

## Problems/Limitations

- Only one seed run so far for E0; multi-seed repeats come with the full
  experiment manifest (later phase).
- `time` column leakage-adjacency noted above needs explicit treatment in
  the final discussion/limitations section.
- MAR/MNAR mechanisms not implemented yet — MCAR only this cycle.
- Missingness/imputation currently scoped to the 7 continuous features;
  the 5 binary clinical flags are not masked (mean/median imputation is
  not a meaningful reconstruction target for a 0/1 flag) — documented in
  `src/config.py`, revisit if KNN/ML imputation later needs them.

## Work in Progress

None mid-flight; this cycle's scope (missingness framework + mean/median
imputation) is complete and committed.

## Next Steps

Phase 11-13: KNN imputation, ML-based imputation (IterativeImputer or
similar), RF after imputation. Then Phase 14+: GA feature selection.
