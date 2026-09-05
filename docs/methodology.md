# Methodology

Consolidated methodology reference. All content here is drawn directly from
`docs/final_report.md` (Sections 8-15) and `docs/progress_report.md`; no new
claim is introduced in this document.

## Dataset and Splitting

The Heart Failure Clinical Records dataset (UCI ID 519, 299 rows, 12
features, target `DEATH_EVENT`) is loaded and validated via
`src/data_loader.py` and `src/config.py`. An 80/20 stratified train/test
split is performed once per seed (`train_test_split_stratified`,
`src/preprocessing.py`), using scikit-learn's stratified `train_test_split`
with the seed passed directly as `random_state` — each of the project's five
repeated seeds draws a fresh train/test partition, not merely fresh model
randomness. Stratified 5-fold cross-validation is used inside the training
split wherever a fitness or tuning signal is required (GA fitness
evaluation).

## Data Leakage Prevention

The held-out test split at no point participates in imputer fitting, GA
fitness evaluation, or any other model-selection decision. This was
re-verified directly in the project's final experiment audit by reading
`src/preprocessing.py` and `src/genetic_selection.py` and confirming no
fitting function accepts or references test-split data; a dedicated
automated test (`tests/test_run_e6.py`) asserts that GA fitness evaluation
only ever receives training-split row indices.

## Missing-Data Mechanisms

Implemented in `src/missingness.py`, at three target rates (10%, 20%, 30%),
scoped to the seven continuous features (the five binary clinical flags are
excluded — mean/median imputation is not a meaningful reconstruction target
for a binary indicator). The target column and identifiers are never masked.

- **MCAR** (Missing Completely at Random): cells masked independently at
  random, no dependence on any variable.
- **MAR** (Missing at Random): masking probability conditioned on other
  *observed* variables, conditioning documented per feature in
  `docs/experiment_log.md`.
- **MNAR** (Missing Not at Random): masking probability conditioned on the
  value being masked itself (or a defensible approximation), assumption
  documented per feature.

For every intentionally masked cell, the true value is preserved before
masking, so imputation error is measured only against genuinely hidden
ground truth. Realised missingness percentages closely matched requested
percentages (e.g. MCAR training-split realised rates: 10.04%, 20.08%,
30.13%). A statistical sanity check (point-biserial correlation, p < 0.05)
confirmed MAR/MNAR masking probabilities actually correlated with their
documented conditioning variables.

## Imputation Methods

Implemented in `src/imputation.py`:

- **Mean imputation** — training-split column mean.
- **Median imputation** — training-split column median.
- **KNN imputation** — k = 5 (scikit-learn default), fit on the training split.
- **IterativeImputer** — scikit-learn's round-robin multivariate imputer, a
  single-imputation, computationally lighter alternative to a full
  multiple-imputation procedure such as MICE or MissForest, a scope-control
  choice appropriate to this project's dataset size and timeframe.

An optional Autoencoder-based imputer was not implemented: at 12 features
and 299 rows, it is not scientifically justified relative to the classical
and iterative methods already implemented.

## Genetic Algorithm (GA) Feature Selection

Implemented in `src/genetic_selection.py`. Each candidate feature subset is
a binary chromosome (1 = selected, 0 = removed); the all-zero chromosome is
disallowed. Tournament selection, single-point crossover, and elitism are
used, with parameters matching the project's specified defaults exactly:

| Parameter | Value |
|---|---|
| Population size | 30 |
| Generations | 30 |
| Crossover probability | 0.8 |
| Tournament size | 3 |
| Elite count | 2 |
| Mutation probability | ≈ 1/12 (reciprocal of chromosome length) |

Fitness = mean 5-fold stratified CV F1 (training split only) minus 0.02 ×
selected-feature ratio. Fitness evaluation uses 50 trees per Random Forest
fit rather than the final model's 200 (CV-F1 ranking is stable well before
200 trees, and fitness is evaluated up to 900 times per GA run); the final
all-features-vs-GA comparison uses the full 200-tree configuration on both
sides. The nine-cell imputation-plus-GA sweep (E6) used a reduced population
of 20 and 15 generations for compute-budget reasons (documented in
`src/config.py`); this reduction did not apply to the complete-data GA runs
(E5).

## Random Forest Prediction

scikit-learn's Random Forest is the sole downstream prediction model
throughout the project. All final reported comparisons use 200 trees.
Accuracy, Precision, Recall, F1-score, and ROC-AUC are recorded for every
run; hyperparameters, seed, and runtime are logged. No hyperparameter was
ever tuned against the held-out test set.

## Metrics

Imputation quality: MAE and RMSE per feature/mechanism/level/seed, on
intentionally masked cells only. Given the wide difference in feature scales
(e.g. `platelets` ~10^4 vs `serum_creatinine` ~1), per-feature values are
reported individually rather than averaged. Prediction quality: Accuracy,
Precision, Recall, F1-score, ROC-AUC on the held-out test split.

## Experimental Setup

| Group | Description |
|---|---|
| E0 | Complete-data Random Forest baseline |
| E1 | Mean/median imputation |
| E2 | KNN imputation |
| E3 | IterativeImputer (ML-based) imputation |
| E4 | Autoencoder (optional) — not attempted, not justified at this scale |
| E5 | GA feature selection (complete data, 5 seeds) |
| E6 | Imputation + GA + Random Forest (9-cell single-seed sweep, 3-cell/3-seed replication) |
| E7 | Literature comparison (contextual only, no cross-dataset head-to-head claims) |

Five seeds — 42, 123, 2026, 7, 99 — were used throughout. The multi-seed
missingness matrix covers 3 mechanisms × 3 levels × 4 imputers × 5 seeds =
180 runs. Rather than a single master `run_all.py` script, the project uses
one independently re-runnable script per experiment group under
`experiments/`, a documented scope-control simplification (see
`docs/final_report.md` Section 20, Limitations).

Full detail, results, and discussion are in `docs/final_report.md`.
