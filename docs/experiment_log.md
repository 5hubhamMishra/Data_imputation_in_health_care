# Experiment Log

## 2026-09-05 — Repository scaffold

Commit: ea34f27

## 2026-09-05 — Dataset discovery and selection

Selected Heart Failure Clinical Records (UCI id 519). See
`docs/dataset_selection.md`. Commit: b47a3f1

## 2026-09-05 — Acquisition, preprocessing, EDA, E0 baseline

- Downloaded dataset from UCI static export, verified shape (299, 13),
  0 missing values, 0 duplicates, class balance 203/96 — matches the
  discovery-phase claim exactly. SHA-256: see
  `data/raw/heart_failure_metadata.json`.
- Added `src/config.py`, `src/data_loader.py`, `src/preprocessing.py`.
- Added `tests/test_data_loader.py` (3 tests, all passing).
- Generated EDA figures: class distribution, feature distributions,
  correlation matrix.
- Ran E0 complete-data RF baseline (seed 42): Accuracy 0.817, F1 0.667,
  ROC-AUC 0.883. Result: `results/metrics/e0_complete_rf_baseline.json`.

## 2026-09-05 — Missingness framework + mean/median imputation

- Added `src/missingness.py` (MCAR masking, ground truth preserved),
  4 passing tests (`tests/test_missingness.py`).
- Ran MCAR masking at 10/20/30% on the training split (seed 42), realized
  10.04/20.08/30.13%. Masked data + ground truth in `data/processed/`,
  summary in `results/tables/missingness_summary.csv`.
- Generated missing-per-feature and missingness-heatmap figures for MCAR
  20% (representative level).
- Added `src/imputation.py` (mean, median), ran E1 imputation experiment,
  MAE/RMSE per feature/level in `results/metrics/e1_mean_median_imputation.csv`.

## 2026-09-05 — KNN/ML imputation and RF-after-imputation

- Refactored `src/imputation.py` onto sklearn `SimpleImputer`/`KNNImputer`/
  `IterativeImputer` behind a `fit_imputer`/`apply_imputer` pair (train-fit,
  test-transform) plus self-contained `impute_*` wrappers for reconstruction
  scoring. Verified numerically identical to the old fillna-based mean/median
  by rerunning E0 and E1 — outputs unchanged.
- Added E2 (KNN, k=5) and E3 (IterativeImputer) reconstruction scoring:
  `results/metrics/e2_knn_imputation.csv`, `e3_iterative_imputation.csv`.
  KNN is consistently the worst reconstructor by MAE/RMSE on this dataset;
  mean/median/iterative are close to each other.
- Added `src/prediction.py` (shared RF train/evaluate helper, reused by E0
  and the new experiment) and `experiments/run_prediction.py`: 4 imputation
  methods x 3 MCAR levels = 12 RF runs, each imputer fit on the masked
  training split only and applied to an independently-masked test split.
  Result: `results/metrics/e1_e2_e3_rf_prediction.csv`. At 10% MCAR,
  KNN-imputed data (Acc 0.817/F1 0.667/ROC-AUC 0.877) nearly matches the E0
  complete-data baseline (0.817/0.667/0.883) despite being the worst
  reconstructor — reconstruction quality and downstream prediction quality
  do not rank the same way here (single-seed observation, not yet
  statistically tested).
- Added `tests/test_imputation.py` (3 tests, including a leakage check that
  a fitted imputer uses train statistics, not test statistics). Full suite:
  10/10 passing.

## 2026-09-05 — MAR and MNAR mechanisms

- Added `mask_mar`/`mask_mnar` to `src/missingness.py`, both reusing a new
  `_weighted_missing_indices` helper (rank-weighted sampling without
  replacement, so missingness probability is a mild monotonic gradient, not
  a deterministic cutoff). Each mechanism covers 3 of the 7 continuous
  features (documented conditioning schemes for each, see module docstring):
  - MAR: `serum_creatinine` ~ `age` (high), `ejection_fraction` ~ `time`
    (low), `serum_sodium` ~ `creatinine_phosphokinase` (high).
  - MNAR: `ejection_fraction` (low), `serum_creatinine` (high),
    `serum_sodium` (low) — each conditioned on its own value.
- Added 8 tests confirming target/protected columns are never masked,
  ground truth is recoverable, actual-vs-requested missingness is close,
  and — the substantive check — that induced missingness actually
  correlates with the documented conditioning variable in the documented
  direction (`pointbiserialr`, p < 0.05 for every key column). Full suite:
  18/18 passing.
- Generalized `experiments/run_missingness.py`, `run_imputation.py`, and
  `run_prediction.py` to loop over MCAR/MAR/MNAR (previously MCAR-only).
  Reran all three: MCAR realized percentages reproduced exactly
  (10.04/20.08/30.13, confirming determinism), MAR/MNAR realized ~10.04/
  20.08/30.13 on their 3 columns. E1/E2/E3 reconstruction metrics and the
  RF-after-imputation comparison now include all 3 mechanisms —
  `results/metrics/e1_mean_median_imputation.csv`, `e2_knn_imputation.csv`,
  `e3_iterative_imputation.csv`, `e1_e2_e3_rf_prediction.csv`.
- **Confound identified and flagged (not fixed this cycle)**: MAR/MNAR
  downstream RF metrics come out *better* than both MCAR and even the E0
  complete-data baseline (e.g. MNAR 10% mean-imputed: Acc 0.850/F1 0.743 vs
  E0's 0.817/0.667). This is not evidence MAR/MNAR are "easier" — it is
  because MAR/MNAR only mask 3 of the 7 continuous features, and critically
  never mask `time`, the single most predictive (leakage-adjacent) column,
  whereas MCAR masks and re-imputes all 7 including `time`. The mechanisms
  are therefore not being compared on equal footing. A fair mechanism
  comparison requires masking the *same* column set under MCAR/MAR/MNAR —
  logged as a required fix before RQ1 analysis relies on this comparison
  (see Problems/Limitations in progress_report.md).

## Cycle: Confound fix + repeated seeds (commit TBD)

- **Fixed the column-set confound**: extended `MAR_CONDITIONING` and
  `MNAR_CONDITIONING` (`src/missingness.py`) from 3 to all 7 continuous
  columns. New conditioning rules added:
  - MAR: `age` ~ `serum_creatinine` (high), `creatinine_phosphokinase` ~
    `ejection_fraction` (low), `platelets` ~ `age` (high), `time` ~
    `serum_sodium` (low).
  - MNAR: `age` (high), `creatinine_phosphokinase` (high), `platelets`
    (low), `time` (low) — each conditioned on its own value.
  - Every new rule ships with a documented clinical rationale (see
    docstring comments in `src/missingness.py`); no self-referential MAR
    conditioning (a column never conditions on itself).
  - Full test suite (18 tests, sanity checks now spanning 7 columns per
    mechanism instead of 3) still passes.
  - Reran `run_missingness.py`, `run_imputation.py`, `run_prediction.py`.
    MCAR realized percentages reproduced exactly (10.04/20.08/30.13,
    confirming determinism); MAR/MNAR now realize the identical
    percentages on all 7 columns, matching MCAR's column coverage exactly.
- **Confound fix did not fully resolve the MAR-vs-MCAR gap**: after the
  fix, MAR still shows higher RF-after-imputation metrics than MCAR/E0 at
  several levels (e.g. MAR 30% + KNN: F1 0.833, ROC-AUC 0.933). Investigated
  via repeated seeds rather than assumed away.
- **Added `experiments/run_repeated_seeds.py`** (master prompt phase 19).
  Threaded `seed` as an overridable parameter through
  `train_test_split_stratified` (`src/preprocessing.py`) and
  `train_evaluate_rf` (`src/prediction.py`), both defaulting to the
  existing project seed (42) so no prior single-seed result changes —
  verified by rerunning `run_baselines.py` with the default seed and
  reproducing E0's original numbers exactly (Acc 0.8167/F1 0.6667/ROC-AUC
  0.8825).
  - E0 across 5 seeds (42/123/2026/7/99): F1 ranges from 0.667 (seed 42) to
    0.857 (seed 123) — std 0.081. Full per-seed table:
    `results/metrics/repeated_seeds_e0.csv`.
  - E1 mean-imputation @ MCAR 20% across the same 5 seeds: F1 std 0.118.
    `results/metrics/repeated_seeds_e1_mcar20.csv`.
  - **Conclusion**: the single-seed MAR-vs-MCAR gap (a few hundredths to
    ~0.17 in F1 depending on level) is within the range of pure seed noise
    observed for E0 alone. The mechanism-comparison question (RQ1) is not
    resolvable from single-seed data on this dataset's 60-row test set —
    it needs the full multi-seed manifest and a paired statistical test
    (sections 29, 33), not a further tweak to the masking scheme.

## 2026-09-05 — GA feature selection (E5) and all-features-vs-GA comparison

- **Implemented** `src/genetic_selection.py`: binary chromosome (1 bit per
  of the 12 features), tournament selection (tournament_size=3),
  single-point crossover (crossover_probability=0.8), bit-flip mutation
  (mutation_probability=1/12), elitism (elite_count=2), population=30,
  generations=30 — all exactly the master prompt's specified defaults
  (section 25). Zero-feature chromosomes are prevented by a repair step
  (flip one random bit on) applied at init and after every
  crossover/mutation.
- Fitness = mean 5-fold stratified CV F1 (computed on the training split
  only) minus `0.02 * selected_feature_ratio`. Lambda=0.02 chosen so a
  full 0-to-1 swing in feature ratio costs at most 0.02 F1 — small
  relative to this dataset's ~0.05-0.1 CV-F1 noise, nudging toward
  parsimony without dominating the signal.
- **Leakage guard**: `run_ga`/`_fitness` accept no test-set argument at
  all — the held-out test set cannot reach GA fitness by construction.
  Verified by `tests/test_genetic_selection.py::test_fitness_and_run_ga_have_no_test_set_parameter`
  (inspects both functions' signatures).
- **Compute-budget deviation from spec, documented**: GA fitness
  evaluation uses `n_estimators=50` (`GA_FITNESS_N_ESTIMATORS` in
  `src/config.py`) instead of the project's standard 200, since fitness is
  evaluated up to population x generations (900) times per GA run and
  CV-F1 ranking between feature subsets is stable well before 200 trees.
  Measured: a single 200-tree/5-fold CV fitness call took ~1.5s on this
  dataset; 50 trees with no `n_jobs` parallelism overhead took ~0.3s — the
  combination made 5 full GA runs (900 evaluations each, deduplicated via
  a per-run fitness cache since elitism/convergence repeat chromosomes)
  complete in one work cycle instead of an estimated 50-75 minutes. The
  final all-features-vs-GA comparison uses the standard 200-tree RF config
  for both variants, so this deviation affects only the search process,
  not the reported comparison numbers.
- Ran GA on **complete (unimputed) training data**, one run per each of
  the 5 established seeds (42/123/2026/7/99) — all 5 completed with real
  saved output (`experiments/run_ga.py` stdout, generation logs in
  memory per run). Convergence plot for seed 42:
  `results/figures/ga_convergence.png`.
- **Feature-selection frequency** across the 5 runs
  (`results/tables/ga_feature_frequency.csv`,
  `results/figures/ga_feature_frequency.png`): `time` and
  `ejection_fraction` selected in all 5 runs (1.0 frequency);
  `serum_creatinine` in 4/5 (0.8); `diabetes` in 3/5 (0.6);
  `serum_sodium`/`platelets`/`anaemia` in 2/5 (0.4) each;
  `creatinine_phosphokinase`/`sex`/`high_blood_pressure`/`smoking` in 1/5
  (0.2) each; `age` in 0/5 — never selected across 5 independent runs.
  Selected feature count per run ranged 4-7 (out of 12).
- **All-features vs GA-selected** (majority-vote subset, frequency ≥0.5:
  time, ejection_fraction, serum_creatinine, diabetes — 4/12 features,
  66.7% reduction), same seed-42 split as E0, same 200-tree RF config,
  held-out test (n=60): all-features Accuracy 0.817 / Precision 0.786 /
  Recall 0.579 / F1 0.667 / ROC-AUC 0.883 (reused from
  `e0_complete_rf_baseline.json`, not rerun) vs GA-selected Accuracy 0.833
  / Precision 0.800 / Recall 0.632 / F1 0.706 / ROC-AUC 0.815
  (`results/metrics/all_features_vs_ga_complete.csv`). Mixed result:
  Accuracy/F1 improved with a much smaller feature set, ROC-AUC dropped.
  Reported as-is per master prompt section 49 (do not force GA to win) —
  not yet a statistically supported claim on one seed.
- Added `tests/test_genetic_selection.py` (5 tests: no zero-feature
  chromosomes from init, repair flips exactly one bit, fitness/run_ga have
  no test-set parameter, run_ga returns a non-empty selection on synthetic
  data, mutation probability matches the configured rate empirically).
  Full suite: **23/23 passing**.
- Not yet done (next cycle): Imputation + GA + RF (E6) — applying GA to
  the MCAR/MAR/MNAR-imputed datasets rather than only complete data; a
  multi-seed all-features-vs-GA comparison.

## Cycle: Imputation + GA + Random Forest (E6)

- Per-cell imputer selection was read from `e1_e2_e3_rf_prediction.csv`
  (`idxmax` on `test_f1` grouped by mechanism/missing_pct), not assumed:
  MAR 10/20/30 → knn; MCAR 10 → knn, MCAR 20/30 → iterative; MNAR 10/20 →
  median, MNAR 30 → iterative.
- `run_ga` (`src/genetic_selection.py`) gained optional
  `population_size`/`generations` parameters, defaulting to the unchanged
  30/30 (E5's already-reported values are untouched). E6 calls it with
  20/15 — measured cost at 30/30 was ~204s/cell, x9 cells > 30 minutes;
  20/15 measured ~90-110s/cell. This only reduces search thoroughness for
  E6's sweep, not E5's complete-data result.
- All 9 cells completed with real, saved metrics — none skipped
  (`results/metrics/e6_imputed_ga_rf.csv`). GA-selected feature counts
  ranged 2-11 (out of 12) across cells.
- 4-way comparison assembled (`results/tables/four_way_comparison.csv`,
  `results/figures/four_way_comparison.png`): complete+RF (E0, constant
  reference), complete+GA+RF (E5, constant reference), imputed+RF (E1-E3,
  per-cell), imputed+GA+RF (E6, per-cell, this cycle).
- Headline: GA improved F1 in 3/9 cells (MCAR 20%, MCAR 30%, MNAR 10%),
  tied in 1/9 (MAR 20%), hurt in 5/9. Consistent with the complete-data GA
  result (also mixed) — no evidence GA reliably helps on this dataset,
  reported as-is per master prompt section 49.
- Added `tests/test_run_e6.py` — patches `run_ga` to record the row
  indices it receives and asserts they equal the training split's indices
  exactly (disjoint from the test split), confirming the E6 pipeline's
  masking/imputation/GA sequence never leaks test rows into GA fitness.
  Full suite: **24/24 passing**.
- Not yet done (next cycle): full multi-seed coverage of E1-E3/E6 (all 9
  cells currently single-seed), paired significance tests (section 33)
  before any RQ conclusion is drawn.
