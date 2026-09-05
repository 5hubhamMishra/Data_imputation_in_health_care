# Methodology Summary

**Split**: 80/20 stratified train/test split (`train_test_split_stratified`,
`src/preprocessing.py`), seed-parameterized (`seed` argument feeds
`train_test_split`'s `random_state` directly) so each of the 5 repeated
seeds draws a *fresh* split, not just fresh model randomness. Stratified
5-fold CV is used inside the training split wherever a fitness/tuning
signal is needed.

**Leakage discipline** (master prompt section 17-18, re-verified in the
final experiment audit): imputers/scalers/GA are fit on the training split
only; the held-out test split never participates in imputer fitting, GA
fitness, or hyperparameter selection. A dedicated leakage test
(`tests/test_run_e6.py` and the GA leakage test) asserts GA fitness only
ever receives training-split row indices.

**Missingness mechanisms** (`src/missingness.py`): MCAR (uniform random
masking), MAR (probability conditioned on other observed variables), MNAR
(probability conditioned on the masked value itself) — each at 10/20/30%,
scoped to the 7 continuous features (the 5 binary clinical flags are not a
meaningful mean/median reconstruction target). MAR/MNAR conditioning
schemes and rationale: `docs/experiment_log.md`. Ground truth for every
intentionally-masked cell is preserved before masking so imputation error
is measurable only on hidden cells, never on naturally-missing data (there
is none in this dataset).

**Imputation methods**: mean, median, KNN (k=5, sklearn default), and
scikit-learn `IterativeImputer` (the project's single-imputation, lighter
alternative to full MICE/MissForest — a documented scope-control choice,
`docs/literature_review.md` §2).

**Prediction model**: Random Forest (scikit-learn), 200 trees for all final
reported comparisons. Accuracy, Precision, Recall, F1, ROC-AUC recorded for
every run.

**GA feature selection** (`src/genetic_selection.py`, master prompt
sections 25-27): binary chromosome (1=selected/0=removed, zero-feature
chromosomes prevented); population=30, generations=30, crossover
probability=0.8, tournament size=3, elite count=2, mutation probability
≈1/12 (the master prompt's exact specified defaults) for the main GA runs
(E5); E6's 9-cell sweep uses a documented reduced 20/15 population/generations
for compute-budget reasons (`src/config.py`). Fitness = mean 5-fold
stratified CV F1 minus a 0.02× feature-ratio penalty, computed on the
training split only; fitness evaluation uses 50 trees instead of the final
model's 200 (documented in `src/config.py`: CV-F1 ranking is stable well
before 200 trees, and fitness is evaluated up to 900 times per GA run). The
final all-features-vs-GA comparison always uses the full 200-tree
configuration on both sides.

**Repeated seeds**: {42, 123, 2026, 7, 99} threaded through the split, RF,
and CV for statistical robustness (master prompt section 29).

**Statistics**: paired t-test + Wilcoxon signed-rank cross-check across the
5 seeds for every comparison family (imputer-vs-imputer, mechanism-vs-
mechanism, complete-vs-imputed). No multiple-comparison correction applied,
by documented choice (`docs/statistical_analysis.md`): at n=5, any standard
correction would make almost every comparison non-significant regardless of
true effect size.
