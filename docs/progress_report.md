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
- **IMPLEMENTED, EXECUTED, VALIDATED**: MAR/MNAR-vs-MCAR column-set confound
  fixed — `MAR_CONDITIONING`/`MNAR_CONDITIONING` extended from 3 to all 7
  continuous columns (including `time`), each with its own documented
  clinical rationale. All 18 tests still pass with the extended column set
  (sanity checks now cover 7 columns per mechanism, not 3). Reconstruction
  and RF-after-imputation metrics regenerated for MAR/MNAR at all 3 levels;
  MCAR reproduced identical realized percentages, confirming determinism.
- **IMPLEMENTED, EXECUTED, VALIDATED**: Repeated-seed aggregation
  (`experiments/run_repeated_seeds.py`, master prompt phase 19) — seeds
  {42, 123, 2026, 7, 99} threaded through the train/test split, RF, and CV
  (`seed` parameter added to `train_test_split_stratified` and
  `train_evaluate_rf`, defaulting to the existing project seed so prior
  single-seed results are unaffected — confirmed by rerunning E0 with the
  default seed and reproducing the original numbers exactly). Covers E0 and
  E1-mean@MCAR20 as the established pattern; full multi-seed coverage of
  every mechanism x level x imputer cell is later work.

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
RMSE for MAR/MNAR are broadly similar to MCAR's values for the same columns.

**Column-coverage confound fix and its result**: after extending MAR/MNAR
to mask all 7 continuous columns (same set as MCAR, including `time`) and
regenerating all metrics, MAR (and to a lesser extent MNAR) *still* show
higher point-estimate RF-after-imputation metrics than MCAR/E0 at several
levels (e.g. MAR 30% + KNN: F1 0.833, ROC-AUC 0.933, both above E0's 0.667/
0.883). This is no longer explained by column coverage — it is most likely
**single-seed noise on a 60-row held-out test set**, not a genuine
mechanism effect: the repeated-seed run below shows E0 alone swings from
F1 0.667 (seed 42) to F1 0.857 (seed 123) — a wider range than the
MAR-vs-MCAR gap being discussed. This is an explicit, honest flag, not a
resolved finding: **no mechanism comparison claim (RQ1) should be drawn
from single-seed numbers**; a proper comparison needs the full multi-seed
manifest and a paired statistical test (master prompt sections 29, 33),
which is later work.

**Repeated-seed results** (`experiments/run_repeated_seeds.py`, 5 seeds:
42/123/2026/7/99):
- E0 complete-data baseline: Accuracy 0.863 ± 0.040, Precision 0.802 ± 0.079,
  Recall 0.768 ± 0.156, F1 0.776 ± 0.081, ROC-AUC 0.918 ± 0.031.
  (`results/metrics/repeated_seeds_e0.csv`)
- E1 mean-imputation @ MCAR 20%: Accuracy 0.823 ± 0.056, Precision 0.737 ±
  0.079, Recall 0.684 ± 0.186, F1 0.702 ± 0.118, ROC-AUC 0.857 ± 0.053.
  (`results/metrics/repeated_seeds_e1_mcar20.csv`)

The std on F1 (0.08-0.12) and recall (0.16-0.19) across just 5 seeds is
large relative to most of the single-seed differences reported earlier in
this document — the clearest evidence yet that this dataset's 60-row test
set makes single-seed rankings unreliable, and multi-seed aggregation (not
yet complete for every cell) is necessary before any RQ conclusion.

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
- `results/metrics/repeated_seeds_e0.csv`
- `results/metrics/repeated_seeds_e1_mcar20.csv`

## Problems/Limitations

- Only one seed run so far for most experiment cells (E1-E3 across all
  mechanisms/levels, RF-after-imputation); the repeated-seed pattern is
  established for E0 and E1@MCAR20 only (std of 0.08-0.12 on F1 across 5
  seeds) — full multi-seed coverage of every cell is still needed before
  any method/mechanism ranking is statistically supported.
- `time` column leakage-adjacency noted above needs explicit treatment in
  the final discussion/limitations section.
- **MAR/MNAR vs MCAR/E0 comparison remains inconclusive on single-seed
  data**: the column-coverage confound is fixed (all mechanisms now mask
  the same 7 columns), but MAR still shows higher point-estimate metrics
  than MCAR/E0 at some levels. Given E0's own 5-seed std on F1 is 0.081,
  this single-seed gap is not distinguishable from noise. Needs the full
  multi-seed manifest + paired significance test before RQ1's mechanism
  comparison can be written up.
- Missingness/imputation currently scoped to the 7 continuous features;
  the 5 binary clinical flags are not masked (mean/median imputation is
  not a meaningful reconstruction target for a 0/1 flag) — documented in
  `src/config.py`.
- KNN's k=5 is sklearn's default, not CV-tuned; IterativeImputer uses
  scikit-learn defaults beyond the seed. Neither is unreasonable, but
  neither is optimized — a fixed, honestly-reported starting point.

## Work in Progress

None mid-flight; this cycle's scope (confound fix + repeated-seeds pattern)
is complete and committed. Full multi-seed coverage of every mechanism x
level x imputer cell remains open for a near-term cycle.

## Next Steps

GA feature selection implementation (master prompt sections 25-27), then
the full Imputation + GA + RF experiment group (E6). Full multi-seed
coverage and paired statistical tests (sections 29, 33) before any RQ1/RQ2
conclusion is finalized in the report draft.
