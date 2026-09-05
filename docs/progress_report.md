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
- **IMPLEMENTED, EXECUTED, VALIDATED**: KNN (E2, k=5) and IterativeImputer
  (E3) added to `src/imputation.py`, same reconstruction scoring as E1 —
  `results/metrics/e2_knn_imputation.csv`, `results/metrics/e3_iterative_imputation.csv`.
- **IMPLEMENTED, EXECUTED, VALIDATED**: `src/prediction.py` — shared
  train/evaluate RF helper, reused by E0 and by the new RF-after-imputation
  experiment (`experiments/run_prediction.py`), which fits each imputer on
  the training split only and `.transform`s a separately-masked held-out
  test split (no leakage), covering all 4 methods x 3 MCAR levels (12 runs)
  — `results/metrics/e1_e2_e3_rf_prediction.csv`. 3 new passing tests
  (`tests/test_imputation.py`), including a leakage check that a fitted
  imputer uses train statistics, not test statistics.
- **IMPLEMENTED, EXECUTED, VALIDATED**: MAR and MNAR missingness mechanisms
  (`src/missingness.py`), each with a documented conditioning scheme over 3
  of the 7 continuous features, 8 passing tests including a statistical
  sanity check that the induced missingness actually correlates with its
  documented conditioning variable (`pointbiserialr`, p<0.05). Full suite:
  18/18 passing. All three experiment scripts generalized to loop over
  MCAR/MAR/MNAR and rerun; MCAR results reproduced exactly, confirming
  determinism.

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
seed 42, 5-fold stratified CV on the training set only. E0 (complete data)
is the reference point. MCAR missingness at 10/20/30% is applied
independently to the training and test splits; each imputer (mean, median,
KNN, IterativeImputer) is fit on the training split only and applied via
`.transform` to the test split before RF evaluation, preventing test-set
leakage into any preprocessing step (master prompt section 17).

## Experiments Completed

- E0: Complete-data RF baseline (1 run, seed 42).
- Missingness framework: MCAR at 10/20/30% on the training split (seed 42).
- E1: Mean and median imputation on each MCAR level, scored on hidden cells.
- E2: KNN (k=5) imputation, same scoring.
- E3: IterativeImputer imputation, same scoring.
- RF-after-imputation: 4 methods (mean/median/knn/iterative) x 3 MCAR
  levels (10/20/30%) = 12 runs, held-out test metrics, compared against E0.
- MAR and MNAR missingness at 10/20/30% (3 conditioned features each), same
  4 imputation methods, same reconstruction scoring and RF-after-imputation
  comparison as MCAR — 24 more RF runs.

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
are the reportable artifact). Mean, median, and IterativeImputer perform similarly at every level (within
noise of each other); KNN (k=5, undocumented/untuned) is consistently the
worst reconstructor on this dataset's feature scales — see
`results/metrics/e2_knn_imputation.csv` vs e1/e3.

RF-after-imputation (`results/metrics/e1_e2_e3_rf_prediction.csv`, single
seed=42 run per cell): at 10% MCAR, KNN-imputed data reaches Accuracy 0.817
/ F1 0.667 / ROC-AUC 0.877 — matching E0's complete-data baseline (0.817 /
0.667 / 0.883) almost exactly, despite being the worst reconstructor by
MAE/RMSE. Mean/median/iterative are slightly behind KNN at 10% (F1
0.53-0.56). All methods degrade as missingness rises to 30% (F1 drops to
0.41-0.44), as expected. This is a single-seed result per cell — not yet a
statistically supported ranking; that requires the multi-seed manifest
(later phase). The disconnect between reconstruction quality (MAE/RMSE) and
downstream prediction quality (F1/ROC-AUC) is worth flagging for RQ2/RQ4:
best imputer by reconstruction error is not necessarily best for prediction.

MAR and MNAR were implemented with documented conditioning schemes (see
`docs/experiment_log.md` for the exact variables/directions) and pass a
statistical sanity check confirming the induced missingness actually
correlates with its conditioning variable as designed. Reconstruction MAE/
RMSE for MAR/MNAR's 3 covered columns are broadly similar to MCAR's values
for the same columns. **However, the RF-after-imputation numbers for MAR/
MNAR should not yet be read as "MAR/MNAR are easier than MCAR"**: MAR/MNAR
only mask 3 of the 7 continuous features and never touch `time` (the most
predictive, leakage-adjacent column), while MCAR masks and re-imputes all 7
including `time`. This confound must be resolved (mask the same column set
across all three mechanisms) before RQ1's mechanism comparison is written
up — see Problems/Limitations.

## GA Status

Not started. Scheduled for a later cycle (master prompt phase 14+),
after the missingness/imputation framework exists.

## Important Tables/Figures

- `results/dataset_candidate_comparison.csv`
- `results/figures/class_distribution.png`
- `results/figures/feature_distributions.png`
- `results/figures/correlation_matrix.png`
- `results/figures/missing_per_feature_{mcar,mar,mnar}20.png`
- `results/figures/missingness_heatmap_{mcar,mar,mnar}20.png`
- `results/metrics/e0_complete_rf_baseline.json`
- `results/tables/missingness_summary.csv`
- `results/metrics/e1_mean_median_imputation.csv`
- `results/metrics/e2_knn_imputation.csv`
- `results/metrics/e3_iterative_imputation.csv`
- `results/metrics/e1_e2_e3_rf_prediction.csv`

## Problems/Limitations

- Only one seed run so far for every experiment (E0-E3, RF-after-imputation);
  multi-seed repeats come with the full experiment manifest (later phase) —
  none of the method rankings above are statistically supported yet.
- `time` column leakage-adjacency noted above needs explicit treatment in
  the final discussion/limitations section.
- **MAR/MNAR vs MCAR comparison is currently confounded**: MAR/MNAR mask
  only 3 columns (never `time`), MCAR masks all 7 (including `time`) — so
  MAR/MNAR's better-looking RF-after-imputation numbers reflect less
  information destroyed, not an easier mechanism. Needs a same-column-set
  rerun before RQ1 conclusions are drawn from this comparison.
- Missingness/imputation currently scoped to the 7 continuous features;
  the 5 binary clinical flags are not masked (mean/median imputation is
  not a meaningful reconstruction target for a 0/1 flag) — documented in
  `src/config.py`.
- KNN's k=5 is sklearn's default, not CV-tuned; IterativeImputer uses
  scikit-learn defaults beyond the seed. Neither is unreasonable, but
  neither is optimized — a fixed, honestly-reported starting point.

## Work in Progress

None mid-flight; this cycle's scope (MAR/MNAR mechanisms) is complete and
committed. The MAR/MNAR-vs-MCAR confound noted above is an open item for a
near-term cycle, not mid-flight work.

## Next Steps

Resolve the MAR/MNAR-vs-MCAR column-set confound (mask the same 3 columns
under all three mechanisms for a fair comparison), then repeated seeds
(Phase 19), feature stability, and Phase 14+: GA feature selection.
