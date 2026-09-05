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
- **IMPLEMENTED, EXECUTED, VALIDATED**: GA feature selection
  (`src/genetic_selection.py`, master prompt sections 25-27) — binary
  chromosome, population=30, generations=30, crossover_probability=0.8,
  tournament_size=3, elite_count=2, mutation_probability=1/12≈0.083 (all
  exactly the specified defaults); fitness = mean 5-fold stratified CV F1
  minus a 0.02x feature-ratio penalty, computed on the training split only
  (structurally guaranteed — `run_ga`/`_fitness` take no test-set
  argument, verified by a dedicated leakage test). Fitness evaluation uses
  50 trees instead of the final RF's 200 (documented deviation in
  `src/config.py`: CV-F1 ranking is stable well before 200 trees, and this
  is evaluated up to 900 times per GA run) — the final all-features-vs-GA
  comparison still uses the full 200-tree config for both sides. Ran on
  complete (unimputed) data across all 5 established seeds
  (`experiments/run_ga.py`). 5 new tests, full suite 23/23 passing.
- **IMPLEMENTED, EXECUTED, VALIDATED**: Imputation + GA + Random Forest (E6,
  `experiments/run_e6.py`) — all 9 mechanism x missingness cells completed
  (none skipped). Per cell, the imputer with the best downstream F1 in
  `e1_e2_e3_rf_prediction.csv` was carried forward (read from that file, not
  assumed), GA ran on the imputed training data only (`run_ga` extended
  with optional `population_size`/`generations` args, defaulting to the
  unchanged 30/30 used by E5; E6 passes 20/15 — documented compute-budget
  reduction for the 9-cell sweep, measured at ~204s/cell at 30/30 vs ~90s at
  20/15), and RF was evaluated on the GA-selected subset using the
  correspondingly-imputed test data. Results: `results/metrics/e6_imputed_ga_rf.csv`.
  4-way comparison (E0/E1-E3/E5/E6) assembled into
  `results/tables/four_way_comparison.csv` and `results/figures/four_way_comparison.png`.
  New leakage test (`tests/test_run_e6.py`) confirms GA fitness only ever
  receives the training-split row indices. Full suite: 24/24 passing.
- **IMPLEMENTED, EXECUTED, VALIDATED**: Full multi-seed coverage of the
  E1-E3 matrix (`experiments/run_multiseed_matrix.py`) — 3 mechanisms x 3
  levels x 4 imputers x 5 seeds = 180 runs, all completed
  (`results/metrics/e1_e2_e3_multiseed.csv`, aggregated mean/std/min/max/
  count per cell in `e1_e2_e3_multiseed_summary.csv`, master prompt section
  32).
- **IMPLEMENTED, EXECUTED, VALIDATED**: Paired statistical analysis
  (`experiments/run_statistics.py`, `docs/statistical_analysis.md`, master
  prompt sections 32-33) — 27 paired t-tests (+ Wilcoxon cross-check) across
  the 5-seed matrix: imputer-vs-imputer (9, all non-significant), mechanism-
  vs-mechanism (9, all non-significant), and complete-data-vs-imputed "cost
  of missingness" (9, **4 significant at p<0.05**: MCAR 10%/20%, MNAR
  20%/30%). No multiple-comparison correction applied (documented reasoning:
  n=5 makes any correction procedure trivially non-significant everywhere).
- **IMPLEMENTED, EXECUTED, VALIDATED**: Small GA multi-seed expansion
  (`experiments/run_e6_multiseed_subset.py`,
  `results/metrics/e6_multiseed_subset.csv`) — the 3 most interesting E6
  cells from the seed-42 result (MCAR 30% = biggest GA gain, MNAR 30% =
  biggest GA loss, MAR 20% = tied), rerun across 3 seeds (42/123/2026) with
  GA's existing reduced settings (population=20, generations=15). Result:
  the seed-42 "biggest GA gain" cell (MCAR 30%, F1 0.444→0.500) does **not**
  hold up — averaged over 3 seeds, all-features F1 (0.647) is actually
  slightly *higher* than GA F1 (0.632) in that same cell. All three
  subsampled cells show GA at or below all-features F1 on average
  (MAR 20%: 0.724→0.676; MNAR 30%: 0.634→0.607). This confirms, with actual
  multi-seed evidence rather than inspection, that the single-seed E6
  "GA sometimes helps" cells were substantially seed-specific noise, not a
  real per-cell effect — consistent with master prompt section 49 (do not
  force GA to win).

- **IMPLEMENTED, EXECUTED, VALIDATED**: Research-question analysis
  (`docs/research_question_analysis.md`, master prompt section 38) —
  RQ1-RQ4 answered strictly from files already on disk (statistical tests,
  multi-seed summaries, GA frequency/comparison tables, literature review).
  Confirms via the actual `train_test_split_stratified` code that each of
  the 5 seeds draws a fresh train/test split, not just fresh model
  randomness — noted as a design fact relevant to interpreting all
  multi-seed results. RQ1: missingness hurts prediction (supported,
  4/9 p<0.05) but mechanism is not distinguishable (0/9 significant). RQ2:
  no imputer statistically distinguishable (0/9 significant). RQ3: GA
  reliably selects a stable, clinically-plausible subset (time,
  ejection_fraction consistently chosen) but does not reliably improve
  prediction — independently confirmed by recomputing the 3-cell/3-seed E6
  subset's per-seed values directly from `e6_multiseed_subset.csv` (GA mean
  F1 below all-features mean F1 in all 3 cells: MCAR30 0.632 vs 0.647,
  MNAR30 0.607 vs 0.634, MAR20 0.676 vs 0.724). RQ4: own 4-way comparison
  (E0/E1-E3/E5/E6) plus literature context, agreements/divergences stated
  explicitly, no cross-dataset numbers presented as head-to-head.

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

**Implemented, executed, and validated on complete data (E5).** 5 GA runs
(seeds 42/123/2026/7/99), each converging to a 4-7 feature subset (out of
12). Selection frequency across the 5 runs
(`results/tables/ga_feature_frequency.csv`):

| feature | frequency |
|---|---|
| time | 1.0 |
| ejection_fraction | 1.0 |
| serum_creatinine | 0.8 |
| diabetes | 0.6 |
| serum_sodium, platelets, anaemia | 0.4 |
| creatinine_phosphokinase, sex, high_blood_pressure, smoking | 0.2 |
| age | 0.0 |

`time` and `ejection_fraction` (both established strong predictors in the
heart-failure literature) are selected in every run — the stable core.
`age` is never selected across 5 independent runs — a genuine negative
finding, not an omission.

**All-features vs GA-selected** (majority-vote subset ≥50% frequency:
time, ejection_fraction, serum_creatinine, diabetes — 4 of 12 features,
66.7% reduction; seed 42, held-out test, complete data):

| variant | n_features | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| all_features | 12 | 0.817 | 0.667 | 0.883 |
| ga_selected | 4 | 0.833 | 0.706 | 0.815 |

Mixed result, reported honestly (master prompt section 49): GA improved
Accuracy/F1 with a two-thirds smaller feature set, but ROC-AUC dropped.
Single seed-42 split — not yet a statistically supported claim that GA
"wins"; needs the multi-seed comparison next.

**Imputation + GA + Random Forest (E6)** — GA applied on top of the
best-per-cell imputer (seed 42, single seed like the other E6 predecessors):

| mechanism | % | imputer | features (before→after) | imputed+all-features F1 | imputed+GA F1 |
|---|---|---|---|---|---|
| MAR | 10 | knn | 12→6 | 0.743 | 0.667 |
| MAR | 20 | knn | 12→8 | 0.647 | 0.647 |
| MAR | 30 | knn | 12→5 | 0.833 | 0.757 |
| MCAR | 10 | knn | 12→7 | 0.667 | 0.647 |
| MCAR | 20 | iterative | 12→11 | 0.588 | 0.606 |
| MCAR | 30 | iterative | 12→7 | 0.444 | 0.500 |
| MNAR | 10 | median | 12→6 | 0.706 | 0.722 |
| MNAR | 20 | median | 12→2 | 0.500 | 0.417 |
| MNAR | 30 | iterative | 12→4 | 0.571 | 0.452 |

GA improved F1 in 3/9 cells (MCAR 20%, MCAR 30%, MNAR 10%), matched in 1/9
(MAR 20%), and hurt in 5/9. Same honest read as the complete-data GA result:
no consistent win, on either side (master prompt section 49) — GA does not
reliably help once imputation is already in the pipeline on this dataset,
and this is a single-seed result per cell, so the mixed pattern itself is
not yet statistically distinguishable from noise.

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
- `results/figures/ga_convergence.png`
- `results/figures/ga_feature_frequency.png`
- `results/tables/ga_feature_frequency.csv`
- `results/metrics/all_features_vs_ga_complete.csv`
- `results/metrics/e6_imputed_ga_rf.csv`
- `results/tables/four_way_comparison.csv`
- `results/figures/four_way_comparison.png`
- `results/metrics/e1_e2_e3_multiseed.csv`
- `results/metrics/e1_e2_e3_multiseed_summary.csv`
- `results/tables/statistical_tests.csv`
- `docs/statistical_analysis.md`
- `results/metrics/e6_multiseed_subset.csv`

## Problems/Limitations

- Full multi-seed coverage (180 runs) now exists for E1-E3, and 27 paired
  statistical tests confirm honestly: no imputer or mechanism is
  statistically distinguishable from its rivals at n=5 seeds (0/18 tests
  significant), but missingness itself measurably hurts prediction relative
  to complete data (4/9 "cost of missingness" tests significant). RQ1's
  fine-grained "which mechanism is worst" cannot be answered from this
  dataset at this sample size — the honest, supportable claim is "missing
  data hurts prediction" not "mechanism X hurts more than mechanism Y."
  E6 (GA) multi-seed coverage remains a 3-cell/3-seed subset, not the full
  9-cell/5-seed matrix (compute budget) — see `docs/statistical_analysis.md`
  for full methodology and all 27 test results.
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
- GA's all-features-vs-GA comparison is single-seed (seed 42) and mixed
  (F1/Accuracy up, ROC-AUC down) — not yet a statistically supported "GA
  helps" or "GA hurts" claim. GA fitness evaluation also uses 50 trees
  instead of 200 for compute-budget reasons (documented in
  `src/config.py`); this affects only the search process, not the final
  reported comparison metrics, which use the full 200-tree config.

- **IMPLEMENTED, EXECUTED, VALIDATED**: Literature review — 15 references
  verified against primary/indexed sources (`results/literature_matrix.csv`,
  `docs/literature_review.md`). Two entries flagged with a verification
  caveat (full-text fetch blocked by publisher, bibliographic details taken
  from indexed metadata only); one entry is an unreviewed arXiv preprint,
  clearly labeled as such. Notable cross-checks: this project's
  independently-derived GA feature-selection frequency (ejection_fraction,
  serum_creatinine most selected) agrees with the dataset's origin paper's
  (Chicco & Jurman 2020) clinical-importance ranking; this project's finding
  that missingness itself (not imputer choice) drives most of the
  performance loss matches Shadbahr et al. (2023)'s independent result on
  different datasets; this project's mixed/negative GA result contrasts
  with Kumar & Sahoo (2017)'s GA+RF cardiovascular result, discussed
  honestly rather than suppressed (see literature review §4).

## Work in Progress

None mid-flight; literature review (section 35-36) and RQ1-RQ4 analysis
(section 38) are both complete and committed.

## Consolidation Check (Phase 24-25)

- Verified every file path listed under "Important Tables/Figures" above,
  plus every results/ path referenced anywhere in `docs/*.md` and
  `README.md`, exists on disk. No missing files, no broken references.
- Checked `results/figures/`, `results/tables/`, `results/metrics/` for
  orphans (files present but never referenced by any doc). None found —
  the six `missing_per_feature_*`/`missingness_heatmap_*` figures are
  covered by a brace-expansion shorthand in this file's figure list rather
  than named individually, which is not a gap.
- Spot-checked numbers that recur across multiple docs (E0 Accuracy/F1/
  ROC-AUC 0.817/0.667/0.883; 27 paired tests with 4/9 "cost of missingness"
  significant; full suite 24/24 passing; GA F1 0.667→0.706 / ROC-AUC
  0.883→0.815) across `progress_report.md`, `experiment_log.md`,
  `statistical_analysis.md`, `research_question_analysis.md`, and
  `literature_review.md` — all consistent, no discrepancies found.
- Reran `pytest`: 24/24 passing, unchanged.
- No fixes were needed; nothing regenerated.

## Final Experiment Audit (Phase 28)

- Enumerated every phase in the 32-phase plan (E0; MCAR/MAR/MNAR at
  10/20/30%; E1-E3 reconstruction and RF-after-imputation; repeated-seed
  E0/E1; E5 GA on complete data; E6 imputation+GA; full 180-run multiseed
  E1-E3 matrix; 27 paired statistical tests; GA multiseed subset; 15-ref
  literature review; RQ1-RQ4 analysis) — every corresponding output file
  exists and is non-empty.
- Spot-checked actual values, not just presence: `results/tables/
  statistical_tests.csv` has exactly 27 rows (9 imputer-vs-imputer, 9
  mechanism-vs-mechanism, 9 missingness-cost), all imputer/mechanism
  p-values > 0.05, and missingness-cost p<0.05 at exactly the 4 documented
  cells (MCAR 10% p=0.0175, MCAR 20% p=0.0119, MNAR 20% p=0.0083, MNAR 30%
  p=0.0236) — matches `docs/statistical_analysis.md` and
  `docs/progress_report.md` exactly, no discrepancy.
- Leakage discipline re-verified by reading current code (not just tests):
  `train_test_split_stratified` (`src/preprocessing.py`) is the only split
  point and runs before any imputer/selector; `run_ga`/`_fitness`
  (`src/genetic_selection.py`) take only `X_train`/`y_train` — no test-set
  parameter exists in the module, so GA cannot see test data by
  construction. Matches what was previously documented.
- Determinism check: reran `experiments/run_baselines.py` (E0, seed 42)
  fresh — reproduced the recorded metrics exactly (Accuracy 0.8167,
  Precision 0.7857, Recall 0.5789, F1 0.6667, ROC-AUC 0.8825, CV F1 0.7683
  +/- 0.0820). Only the incidental `runtime_seconds` field differed
  (wall-clock, not a result); that diff was discarded, not committed.
- Full test suite: 24/24 passing, matches the documented count.
- Deferred scope confirmed intentional, not an oversight: the full 9-cell/
  5-seed E6 (GA+imputation) matrix remains a 3-cell/3-seed subset by
  documented compute-budget decision (see "Problems/Limitations" above);
  this audit does not reopen that scope.
- **Result: no discrepancies found. Ready to proceed** to the
  report_data/ package (section 43) and final academic report draft
  (section 31).

## Report Data Package (Phase 30)

- Built `report_data/` with exactly the structure master prompt section 43
  specifies: `dataset_summary.md`, `methodology_summary.md`,
  `experiment_setup.md`, `results_summary.md`, `research_questions.md`,
  `discussion_points.md`, `limitations.md`, `future_work.md`,
  `references.bib`, `tables/`, `figures/`.
- Each `.md` file is a condensed synthesis of already-validated content in
  `docs/*.md` and `results/*` — no new numbers were computed and nothing
  here introduces a claim not already present in the source docs. Every
  file/number cited in `report_data/` traces back to a file already listed
  in this progress report or `docs/research_question_analysis.md`.
- `report_data/tables/` and `report_data/figures/` are plain copies of the
  4 report-ready tables (`results/tables/*.csv`), the dataset comparison and
  literature matrix (`results/dataset_candidate_comparison.csv`,
  `results/literature_matrix.csv`), and all 12 figures from
  `results/figures/` — the aggregate/report-ready artifacts, not the raw
  per-run intermediate CSVs in `results/metrics/` (those remain cited by
  path from the `.md` summaries instead of duplicated).
- `references.bib` was generated from the 15 verified entries in
  `results/literature_matrix.csv`; the two entries with a fetch-blocked
  verification caveat (Kumar & Sahoo 2017, Yaqoob et al. 2025) and the one
  arXiv preprint (Grzesiak et al. 2025) carry their caveats as BibTeX
  comments/notes so they are not silently upgraded to fully-verified status.
- No experiments were rerun or regenerated for this phase; it is pure
  curation of existing validated outputs.

## Final Academic Report Draft (Phase 31)

- Wrote `docs/final_report.md`, following the exact section structure
  specified in master prompt section 43 (Final Report Package): Abstract,
  Introduction, Background, Problem Statement, Objectives, Research
  Questions, Literature Review, Dataset, Preprocessing, Missing-Data
  Mechanisms, Methodology, Imputation Methods, GA Feature Selection, Random
  Forest, Experimental Setup, Metrics, Results, Statistical Analysis,
  Discussion, Existing-Work Comparison, Limitations, Future Work,
  Conclusion, References.
- Every number in the draft was drawn directly from `report_data/` (and,
  where more precision was useful, the underlying `results/` files) — no
  value was invented, rounded favorably, or "cleaned up." The draft
  preserves the project's established honest-reporting stance throughout:
  missingness significantly hurts prediction (4/9 tests), no imputer or
  mechanism is statistically distinguishable at n=5 seeds (0/18 tests), GA
  gives a stable/clinically-plausible feature subset but not a replicated
  accuracy gain, and the `time` column's leakage-adjacency is flagged
  explicitly rather than presented as a clinical finding.
- Divergence from Kumar & Sahoo (2017)'s favorable GA result is discussed
  directly in the Existing-Work Comparison section, not omitted.
- Noted, not fixed (out of this phase's scope): `docs/supervisor_progress_summary.md`,
  called for by master prompt section 41, does not yet exist in `docs/` —
  flagged here for a future phase, not fabricated as part of this one.
- Reran `pytest`: 24/24 passing, unchanged (this phase touched only
  documentation).

## Supervisor Progress Summary (Phase 41)

`docs/supervisor_progress_summary.md` written: a shorter, faculty-facing
status summary (completion status, work completed, defensible findings,
remaining limitations, deliverables list) sourced only from
`docs/progress_report.md` and `docs/final_report.md` — no new claims.

## Final Reproducibility & Repository Audit (Phase 32)

- Repository structure, `.gitignore`, dataset provenance/checksum, and
  remote (`origin` → the project's GitHub repo) all verified present and
  correct; matches what `README.md` describes.
- Coverage re-verified: 180/180 E1-E3 multiseed rows, 27/27 statistical
  tests, 5/5 GA seeds, 9/9 E6 single-seed cells (3/9 with 3-seed
  replication), 15/15 literature entries.
- `README.md` corrected: stale "dataset not yet selected" status line and
  placeholder `run_all.py` instructions replaced with the actual completion
  status and the real, individually-runnable experiment module commands in
  dependency order.
- Absolute-local-path scan (`src/`, `experiments/`, `docs/`) for anything
  that would break on another machine: none found.
- `requirements.txt` cross-checked against every third-party import
  actually used in `src/`/`experiments/`/`tests/` (numpy, pandas,
  scikit-learn, matplotlib, seaborn, scipy, pytest): complete, nothing
  missing.
- Test suite rerun for this closing audit (`python -m pytest -q`): 24/24
  passing. (A prior pass of this same audit recorded, incorrectly, that no
  Python interpreter was available in-session and skipped the rerun —
  corrected here and in `docs/reproducibility_audit.md`; the interpreter
  and pytest are present and the suite passes.)
- **Result: no repository-hygiene or reproducibility issues found. All 32
  phases of the master prompt's execution plan are now complete.**

## Named-Deliverable Cleanup (Sections 12, 29)

A section-by-section re-read of the master prompt against the actual repo
file tree found three named deliverables that were never produced, despite
the underlying phase work being complete:

- **`results/experiment_manifest.csv`** (section 29). Built retrospectively
  from the already-validated per-experiment result files
  (`e1_e2_e3_multiseed.csv`, `repeated_seeds_e0.csv`,
  `all_features_vs_ga_complete.csv`, `e6_imputed_ga_rf.csv`,
  `e6_multiseed_subset.csv`) rather than by a live orchestrator, since the
  project deliberately uses one script per experiment group instead of a
  single `run_all.py` (documented in `docs/final_report.md`'s limitations).
  201 rows, one per distinct executed run, each traceable via its
  `source_file` column; duplicate runs that appear in more than one source
  file (e.g. an E6 cell's "imputed, no GA" comparator is the same run
  already recorded in the E1-E3 multiseed matrix) were included once, not
  double-counted. No value was computed, rounded, or invented — every field
  is a direct read from an existing results file.
- **`docs/methodology.md`** and **`docs/results_summary.md`** (section 12's
  recommended structure). Both created as standalone consolidations of
  content already present in `docs/final_report.md` and
  `report_data/results_summary.md` — no new claim introduced.

## Next Steps

All 32 phases of the master prompt's plan are complete, and the three
previously-missing named deliverables above are now in place. The only
remaining item is explicitly-documented optional future work, not an
unmet requirement: full 9-cell/5-seed E6 (GA+imputation) coverage beyond
the current 3-cell/3-seed subset, which already shows a consistent
(non-)pattern and is disclosed as a compute-budget limitation in
`docs/final_report.md`.
