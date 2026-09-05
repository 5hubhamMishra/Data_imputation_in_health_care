# Healthcare Data Imputation Using Machine Learning with Genetic Algorithm-Based Feature Selection

## Abstract

Missing data is a persistent obstacle in healthcare machine learning, where clinical
records are frequently incomplete due to irregular data collection, patient dropout, or
measurement cost. This project investigates how missing data affects downstream
prediction performance on a clinical dataset, compares four imputation methods (mean,
median, k-nearest neighbours, and an iterative multivariate imputer), and evaluates
whether Genetic Algorithm (GA)-based feature selection improves Random Forest
prediction after imputation. Using the Heart Failure Clinical Records dataset (UCI, 299
patients, 12 features, target `DEATH_EVENT`), missingness was simulated under three
mechanisms — Missing Completely at Random (MCAR), Missing at Random (MAR), and Missing
Not at Random (MNAR) — at 10%, 20%, and 30% rates, with ground truth preserved for every
intentionally masked cell. Across a 180-run multi-seed matrix (3 mechanisms x 3 levels x
4 imputers x 5 seeds) and 27 paired statistical tests, missingness itself was found to
significantly reduce prediction performance relative to complete data in 4 of 9
complete-vs-imputed comparisons (p < 0.05), while no imputation method or missingness
mechanism was statistically distinguishable from any other at this sample size (0 of 18
comparisons significant). GA feature selection reliably and reproducibly selected a
small, clinically plausible feature subset (`time`, `ejection_fraction` chosen in 5 of 5
independent runs) but did not reliably improve prediction accuracy once replicated
across multiple seeds. These results are reported without adjustment to favour any
predetermined conclusion: the project's central, statistically defensible finding is
that the presence of missing data matters more than the specific mechanism or imputation
method used to address it, and that GA's genuine contribution on this dataset is
principled feature reduction rather than an accuracy gain.

## 1. Introduction

Electronic health records and clinical study datasets are rarely complete. Values go
unrecorded because a test was not ordered, a patient missed a follow-up visit, or a
field was left blank during data entry. How a predictive model handles this missingness
— whether through discarding incomplete records, imputing plausible replacement values,
or restricting itself to a robust subset of features — directly affects the reliability
of any downstream clinical decision-support tool. This project studies that problem
empirically on a single, well-characterised clinical dataset, using a controlled
experimental design in which missingness is deliberately introduced so that the true
values remain known and imputation error can be measured directly, rather than relying
on a dataset's uncontrolled, unverifiable natural missingness.

## 2. Background

Three formal mechanisms describe how data can go missing (Rubin, 1976). Missing
Completely at Random (MCAR) means the probability of a value being missing is unrelated
to any observed or unobserved variable. Missing at Random (MAR) means the probability of
missingness depends on other *observed* variables. Missing Not at Random (MNAR) means
the probability of missingness depends on the (unobserved) value itself. These
mechanisms have different practical implications: simple imputation methods that assume
MCAR can introduce bias when the true mechanism is MAR or MNAR. Classical imputation
approaches range from simple univariate statistics (mean/median substitution) to
similarity-based methods (k-nearest neighbours) to fully multivariate, iterative
approaches such as MICE (van Buuren & Groothuis-Oudshoorn, 2011) and MissForest
(Stekhoven & Buhlmann, 2012). Feature selection via Genetic Algorithms has a long
history in applied machine learning (Leardi, Boggia, & Terrile, 1992) as a
population-based wrapper method that can search a combinatorial feature space more
effectively than exhaustive or greedy approaches, at the cost of a stochastic,
CV-fitness-driven search that can overfit small training sets.

## 3. Problem Statement

Two related problems are addressed. First, the effect of missing data on downstream
prediction is often reported inconsistently across the literature because studies vary
in dataset, missingness mechanism, missingness rate, and imputation method
simultaneously, making it difficult to isolate which factor drives observed performance
changes. Second, feature selection methods such as GA are frequently proposed as a way
to both reduce dimensionality and improve accuracy, but claims of GA "improving"
prediction are sometimes drawn from a single train/test split, which cannot distinguish
a genuine effect from split-specific noise. This project addresses both problems using a
single fixed dataset, a controlled missingness framework with known ground truth, and a
repeated multi-seed design with paired statistical testing, so that any claimed effect —
of missingness, of imputer choice, or of GA — can be checked for replication rather than
accepted from a single run.

## 4. Objectives

1. Quantify how missingness (across mechanism and rate) affects Random Forest
   prediction performance relative to a complete-data baseline.
2. Compare four imputation methods (mean, median, KNN, IterativeImputer) on both
   reconstruction accuracy and downstream prediction performance.
3. Determine whether GA-based feature selection improves prediction performance, or
   preserves comparable performance with fewer features, both on complete data and
   after imputation.
4. Situate the project's own results against selected published work on imputation and
   GA-based feature selection in healthcare prediction, without presenting
   different-dataset results as direct head-to-head comparisons.

## 5. Research Questions

- **RQ1.** How does missing data affect healthcare prediction performance?
- **RQ2.** Which selected imputation method performs best for the chosen dataset?
- **RQ3.** Can GA-based feature selection improve prediction performance, or preserve
  comparable performance using fewer features?
- **RQ4.** How does Imputation + GA + Random Forest compare with selected baselines and
  existing research?

## 6. Literature Review

Fifteen references were identified and verified against primary or indexed
bibliographic sources (`results/literature_matrix.csv`; full discussion in
`docs/literature_review.md`). Rubin (1976) provides the foundational MCAR/MAR/MNAR
taxonomy used throughout this project. Van Buuren & Groothuis-Oudshoorn (2011) and
Stekhoven & Buhlmann (2012) describe the multivariate iterative imputation family that
motivates this project's IterativeImputer method, while Troyanskaya et al. (2001)
established KNN imputation in a biological data context. Breiman (2001) is the
foundational Random Forest reference underlying the prediction model used throughout.
Leardi, Boggia, & Terrile (1992) is a foundational reference for GA-based feature
selection. Demsar (2006) informed this project's use of paired statistical tests rather
than a single point-estimate comparison across methods.

Among recent clinical-imputation studies, Shadbahr et al. (2023) found, across several
independent datasets, that the *degree* of missingness — not the specific mechanism or
imputer — was the dominant driver of downstream classifier degradation; this project's
own RQ1 result (Section 12) independently arrives at the same qualitative conclusion on
a different dataset. Ren et al. (2024) systematically reviewed machine-learning-based
imputation in electronic health records and reported no single imputer that
consistently dominates across studies, consistent with this project's RQ2 finding. Aracri
et al. (2025) compared missing-data imputation methods for a dementia classification
task and likewise found no single method statistically dominant. Chicco & Jurman (2020),
the origin study for this project's dataset, identified serum creatinine and ejection
fraction as the two most predictive features for heart failure survival using an
independent feature-importance analysis; this project's GA feature-selection result
(Section 15) selects `ejection_fraction` in every run and `serum_creatinine` in 4 of 5
runs, an independent cross-check that agrees with the origin paper without having used
its analysis. Kumar & Sahoo (2017) reported a favourable GA+Naive-Bayes result for a
cardiovascular classification task; this project's own GA+Random-Forest result
(Section 16) diverges from that finding, and this divergence is discussed directly
rather than omitted (Section 20). Pudjihartono et al. (2022) reviewed feature-selection
methods for disease-risk prediction and discusses hybrid filter-then-wrapper approaches
as a way to reduce a wrapper method's overfitting risk on small training sets, which
motivates this project's future-work discussion of GA overfitting (Section 21). Two
entries — Kumar & Sahoo (2017) and Yaqoob et al. (2025) — carry a verification caveat:
their full text could not be retrieved (HTTP 403 from the publisher), so bibliographic
details and reported findings are drawn from indexed metadata only. Grzesiak et al.
(2025) is cited as an unreviewed arXiv preprint and is labelled as such. Yaqoob et al.
(2025) itself uses Seagull Optimization, not a genetic algorithm, and is cited only for
general feature-selection context, not as a GA comparison point.

## 7. Dataset

The **Heart Failure Clinical Records** dataset (UCI Machine Learning Repository, ID 519,
CC BY 4.0 license) was selected as the project's single primary dataset after comparison
against four other candidates — Chronic Kidney Disease, Heart Disease (Cleveland),
Indian Liver Patient Dataset, and Cervical Cancer Risk Factors — scored across twelve
criteria including healthcare relevance, missing-data suitability, sample size, class
balance, and licensing (`results/dataset_candidate_comparison.csv`,
`docs/dataset_selection.md`). It contains 299 patient records, 12 features, and a binary
target, `DEATH_EVENT`. The dataset is fully observed: zero missing values and zero
duplicate rows were found on integrity checking, and its SHA-256 checksum and full
provenance are recorded in `data/raw/heart_failure_metadata.json`. The class distribution
is moderately imbalanced: 203 survivors (0) and 96 deaths (1), approximately 68%/32%.

This dataset was selected specifically because its complete, natural state allows the
complete-data Random Forest baseline to use all 299 rows without any natural missing
values confounding the analysis, and because every cell that is later intentionally
masked for the controlled MCAR/MAR/MNAR experiments has a known, verifiable ground-truth
value — the cleanest possible setup for measuring imputation error directly. Against
this benefit, the dataset's twelve features are fewer than some rejected candidates (for
example, Chronic Kidney Disease's twenty-four), which makes GA's absolute feature-count
reduction look smaller in raw terms; this trade-off was judged acceptable given that a
clean, unconfounded experimental design was prioritised over a larger but less
controlled feature space.

One feature, `time` (days of follow-up before the recorded outcome), requires explicit
methodological comment: in survival-style clinical data, a short follow-up interval
correlates with the death event almost by construction, since patients who die are, by
definition, not followed further. `time` was retained in the dataset, since removing it
would itself be an undocumented methodological choice not called for by any experimental
requirement, but its outsized influence on Random Forest performance is treated
throughout this report as a modelling caveat rather than a genuine clinical discovery
(see Section 20).

## 8. Preprocessing

Data loading, target detection, and configuration are centralised in `src/data_loader.py`
and `src/config.py`. An 80/20 stratified train/test split is performed once per seed
(`train_test_split_stratified`, `src/preprocessing.py`), using scikit-learn's stratified
`train_test_split` with the seed passed directly as `random_state`; this means each of
the project's five repeated seeds draws a genuinely fresh train/test partition, not
merely a fresh source of model randomness. Stratified 5-fold cross-validation is used
inside the training split wherever a fitness or tuning signal is required (GA fitness
evaluation, in particular). The held-out test split at no point participates in imputer
fitting, GA fitness evaluation, or any other model-selection decision, consistent with a
strict train-then-transform discipline (Section 10).

## 9. Missing-Data Mechanisms

Three missingness mechanisms were implemented in `src/missingness.py`:

- **MCAR**: eligible cells are masked independently at random, with no dependence on any
  observed or unobserved variable.
- **MAR**: the probability of a cell being masked is conditioned on other *observed*
  variables, with the exact conditioning variables and direction documented per feature
  in `docs/experiment_log.md`.
- **MNAR**: the probability of a cell being masked is conditioned on the value being
  masked itself (or a defensible approximation), again with the exact assumption
  documented per feature.

All three mechanisms were applied at three target rates — 10%, 20%, and 30% — and scoped
to the seven continuous features in the dataset; the five binary clinical flags were
excluded from masking, since mean/median imputation is not a meaningful reconstruction
target for a binary indicator. The target column and any identifier columns were never
masked. For every intentionally masked cell, the true value was preserved before masking
so that imputation error could later be measured only against genuinely hidden
ground-truth values — no cell's imputation error is ever computed against a naturally
missing value, since this dataset has none. Realised missingness percentages closely
matched the requested percentages throughout (`results/tables/missingness_summary.csv`);
for example, the MCAR training-split realised rates were 10.04%, 20.08%, and 30.13%
against requested 10/20/30%. A dedicated statistical sanity check confirmed that the MAR
and MNAR masking probabilities actually correlated with their documented conditioning
variables as designed (point-biserial correlation, p < 0.05).

## 10. Methodology

The experimental pipeline follows a fixed conceptual flow: raw data is loaded and
validated, split into training and test partitions, missingness is introduced onto each
partition independently, imputers are fit on the training partition only and then used
to transform both partitions, GA feature selection (where used) operates exclusively on
training-partition cross-validation, and the final Random Forest model is trained on the
(possibly imputed, possibly feature-selected) training data and evaluated once on the
untouched held-out test partition. This split-before-fit discipline was re-verified
directly in the project's final experiment audit by re-reading the relevant source code
(`src/preprocessing.py`, `src/genetic_selection.py`) and confirming that no fitting
function accepts or references test-split data; a dedicated automated leakage test
additionally asserts that GA fitness evaluation only ever receives training-split row
indices (`tests/test_run_e6.py`). No hyperparameter or method-selection decision was
ever informed by the held-out test set.

## 11. Imputation Methods

Four imputation methods were implemented in `src/imputation.py`:

- **Mean imputation**: replaces a missing value with the training-split mean of its
  column.
- **Median imputation**: replaces a missing value with the training-split median of its
  column.
- **K-nearest neighbours (KNN) imputation**: imputes a missing value from the average of
  its k = 5 nearest neighbours (scikit-learn default), fit on the training split.
- **IterativeImputer**: scikit-learn's round-robin multivariate imputer, used here as a
  single-imputation, computationally lighter alternative to a full multiple-imputation
  procedure such as MICE or MissForest — a deliberate scope-control choice appropriate
  to this project's dataset size and single-semester timeframe (`docs/literature_review.md`).

An optional Autoencoder-based imputer (permitted by the project's execution plan as an
optional method) was not implemented: at 12 features and 299 rows, an autoencoder is not
scientifically justified relative to the classical and iterative methods already
implemented, which are sufficient to answer RQ1 and RQ2 on this dataset's scale.

## 12. GA Feature Selection

Genetic Algorithm feature selection was implemented in `src/genetic_selection.py`. Each
candidate feature subset is represented as a binary chromosome (1 = feature selected, 0
= feature removed), with the all-zero chromosome disallowed. The GA used tournament
selection, single-point crossover, and elitism, with the following parameters, matching
the project's specified defaults exactly: population size 30, 30 generations, crossover
probability 0.8, tournament size 3, elite count 2, and mutation probability
approximately 1/12 (the reciprocal of the twelve-feature chromosome length). Fitness was
defined as the mean 5-fold stratified cross-validation F1 score, computed on the
training split only, minus a small feature-count penalty of 0.02 multiplied by the
selected-feature ratio, discouraging unnecessarily large subsets without dominating the
accuracy term. Fitness evaluation used 50 trees per Random Forest fit rather than the
final model's 200, a documented cost-reduction choice justified by the observation that
CV-F1 ranking between candidate subsets is stable well before 200 trees, and because
fitness is evaluated up to 900 times in a single GA run; the final all-features-versus-GA
comparison uses the full 200-tree configuration on both sides, so this shortcut affects
only the search process and not the reported comparison numbers. The nine-cell
imputation-plus-GA sweep (E6) used a reduced population of 20 and 15 generations for
compute-budget reasons; this reduction is documented in `src/config.py` and did not
apply to the complete-data GA runs (E5).

## 13. Random Forest

Random Forest (scikit-learn) was used as the sole downstream prediction model throughout
the project, per the project's scope-control decision to keep the study to one primary
classifier. All final reported comparisons use 200 trees. Accuracy, Precision, Recall,
F1-score, and ROC-AUC were recorded for every run; hyperparameters, seed, and runtime
were logged for every result. No hyperparameter was ever tuned against the held-out test
set.

## 14. Experimental Setup

The project's experiment groups (Section 28 of the governing execution plan) are
summarised below, all implemented, executed, and validated:

| Group | Description | Status |
|---|---|---|
| E0 | Complete-data Random Forest baseline | Done — single seed and 5-seed repeat |
| E1 | Mean/median imputation | Done — MCAR/MAR/MNAR x 10/20/30%, multi-seed |
| E2 | KNN imputation | Done — same coverage as E1 |
| E3 | IterativeImputer (ML-based) imputation | Done — same coverage as E1 |
| E4 | Autoencoder (optional) | Not attempted — not scientifically justified at this dataset's scale |
| E5 | GA feature selection | Done — 5 seeds on complete data |
| E6 | Imputation + GA + Random Forest | Done — full 9-cell single-seed sweep, plus a 3-cell/3-seed multi-seed replication subset |
| E7 | Literature comparison | Done — contextual only, no cross-dataset head-to-head claims |

The multi-seed missingness matrix covers 3 mechanisms (MCAR/MAR/MNAR) x 3 levels
(10/20/30%) x 4 imputers (mean/median/KNN/iterative) x 5 seeds, for 180 total runs
(`results/metrics/e1_e2_e3_multiseed.csv`, aggregated in
`e1_e2_e3_multiseed_summary.csv`). The GA matrix consists of 5 independent complete-data
runs (E5, one per seed) and a 9-cell single-seed sweep after imputation (E6), with the
three most informative E6 cells (largest apparent GA gain, largest apparent GA loss, and
a near-tie at the single seed) replicated across 3 seeds
(`results/metrics/e6_multiseed_subset.csv`). Full 9-cell/5-seed E6 coverage was not
pursued, for compute-budget reasons, since the 3-cell subset already demonstrates a
consistent pattern (Section 16); this is documented as an accepted, non-blocking
limitation rather than a silent omission. Five seeds — 42, 123, 2026, 7, and 99 — were
used throughout, matching the project's specified defaults. Rather than a single master
`run_all.py` script, the project uses one independently re-runnable script per experiment
group under `experiments/`, a deliberate scope-control simplification for a
single-dataset, workstation-scale project, documented rather than silently deviating
from the suggested execution pattern.

## 15. Metrics

Imputation quality was measured on intentionally masked cells only, using Mean Absolute
Error (MAE) and Root Mean Squared Error (RMSE) per feature, mechanism, missingness
level, and seed. Prediction quality was measured using Accuracy, Precision, Recall,
F1-score, and ROC-AUC on the held-out test split. Given the wide difference in feature
scales (for example, `platelets` is on the order of 10^4 while `serum_creatinine` is on
the order of 1), per-feature MAE/RMSE values are reported individually rather than
averaged across features, since a single cross-feature average would not be analytically
meaningful.

## 16. Results

**Complete-data baseline (E0).** Across 5 seeds, the complete-data Random Forest
baseline achieved Accuracy 0.863 ± 0.040, Precision 0.802 ± 0.079, Recall 0.768 ± 0.156,
F1 0.776 ± 0.081, and ROC-AUC 0.918 ± 0.031 (`results/metrics/repeated_seeds_e0.csv`).
The single-seed (seed 42) held-out result was F1 0.667, ROC-AUC 0.883 — the gap between
this single-seed number and the 5-seed mean illustrates directly why no single-seed
result is treated as a final claim anywhere in this project.

**Imputation reconstruction quality.** Mean, median, and IterativeImputer performed
similarly to one another at every missingness level
(`results/metrics/e1_mean_median_imputation.csv`, `e3_iterative_imputation.csv`). KNN
(k = 5, untuned) was consistently the worst reconstructor by MAE/RMSE across this
dataset's feature scales (`results/metrics/e2_knn_imputation.csv`).

**Prediction after imputation, multi-seed.** Across the full 180-run multi-seed matrix
(`results/metrics/e1_e2_e3_multiseed_summary.csv`), F1 ranged approximately from 0.615
to 0.742 across the nine mechanism-by-level cells, degrading as missingness rose from
10% to 30%, as expected. Reconstruction quality and downstream prediction quality did
not track one another: KNN, the worst reconstructor by MAE/RMSE, was sometimes
competitive with or better than the other imputers on downstream F1 — a disconnect
discussed further in Section 20.

**Statistical tests.** Twenty-seven paired comparisons were run across the 5-seed matrix
(paired t-test with a Wilcoxon signed-rank cross-check;
`results/tables/statistical_tests.csv`, full methodology in
`docs/statistical_analysis.md`): 9 imputer-versus-imputer comparisons, 9
mechanism-versus-mechanism comparisons, and 9 complete-data-versus-imputed-data
comparisons. Of these, 4 of 9 complete-versus-imputed comparisons were significant at
p < 0.05 (MCAR 10%, MCAR 20%, MNAR 20%, MNAR 30%), while 0 of 9 imputer-versus-imputer
and 0 of 9 mechanism-versus-mechanism comparisons reached significance. No
multiple-comparison correction was applied, a documented choice: at n = 5 seeds, any
standard correction procedure would push the significance threshold low enough that
almost no comparison could ever reach it, regardless of true effect size; individual
p-values are read as exploratory rather than confirmatory.

**GA feature selection (E5).** Across 5 independent GA runs on complete data, each
converging to a 4-to-7-feature subset out of 12, `time` and `ejection_fraction` were
selected in every run (frequency 1.0), `serum_creatinine` in 4 of 5 runs (0.8),
`diabetes` in 3 of 5 (0.6), and `age` in 0 of 5 runs — a genuine negative finding, not an
omission (`results/tables/ga_feature_frequency.csv`, `figures/ga_feature_frequency.png`,
`figures/ga_convergence.png`).

**All-features versus GA-selected (complete data, seed 42).** Using the majority-vote
subset (features selected in at least 50% of runs: `time`, `ejection_fraction`,
`serum_creatinine`, `diabetes` — a 4-of-12, 66.7% reduction), Accuracy improved from
0.817 to 0.833 and F1 from 0.667 to 0.706, while ROC-AUC fell from 0.883 to 0.815
(`results/metrics/all_features_vs_ga_complete.csv`) — a mixed single-seed result, not
presented as an unqualified GA win.

**Imputation + GA + Random Forest (E6).** Across the nine mechanism-by-level cells at
seed 42, GA improved F1 in 3 of 9 cells, matched it in 1 of 9, and reduced it in 5 of 9
(`results/metrics/e6_imputed_ga_rf.csv`, `results/tables/four_way_comparison.csv`,
`figures/four_way_comparison.png`). The three most informative cells from this
single-seed sweep — the largest apparent GA gain (MCAR 30%), the largest apparent GA
loss (MNAR 30%), and a near-tie (MAR 20%) — were replicated across 3 seeds
(`results/metrics/e6_multiseed_subset.csv`). Averaged over those 3 seeds, GA F1 was at or
below all-features F1 in every one of the three cells (MCAR 30%: 0.632 versus 0.647;
MAR 20%: 0.676 versus 0.724; MNAR 30%: 0.607 versus 0.634) — the single-seed "GA gain" at
MCAR 30% did not replicate.

## 17. Statistical Analysis

The paired statistical testing strategy (Section 16; full detail in
`docs/statistical_analysis.md`) used a paired t-test as the primary test, with a
Wilcoxon signed-rank test run as a non-parametric cross-check for each of the 27
comparisons, since the same 5 seeds are paired across the two conditions being compared
in each case. The result pattern was consistent between the two tests. The one family of
comparisons with any statistically significant results was complete-data-versus-imputed
data (4 of 9 significant), supporting the conclusion that missingness itself, not the
specific mechanism or imputer used to address it, is the dominant, measurable driver of
performance loss in this project's data. Neither the imputer-versus-imputer family nor
the mechanism-versus-mechanism family produced a single significant result at n = 5
seeds; this is interpreted throughout the report as "not distinguishable from noise at
this sample size," not as evidence of no real difference, since the standard deviation
of the complete-data baseline's own F1 across seeds (0.081) exceeds most of the
point-estimate gaps being tested between imputers or mechanisms.

## 18. Discussion

The project's central, statistically supported finding is that **missingness itself
measurably degrades prediction performance relative to complete data**, independent of
which mechanism produced it or which method was used to address it. This is the single
claim in this project backed by a majority-significant test family (4 of 9), and it
independently corroborates Shadbahr et al. (2023)'s finding, obtained on different
datasets, that missingness rate rather than mechanism or imputer choice is the dominant
driver of downstream classifier degradation.

A second notable pattern is the **disconnect between reconstruction quality and
downstream prediction quality**: KNN was consistently the worst imputer by MAE/RMSE, yet
was sometimes competitive with or better than the other imputers on downstream F1. A
plausible explanation is that Random Forest, as an ensemble of decision trees splitting
on threshold values, is comparatively robust to moderate feature-level noise in ways
that a direct reconstruction-error metric does not capture; MAE/RMSE and downstream
classification accuracy are simply answering different questions about imputation
quality.

Third, **GA's demonstrated contribution on this dataset is feature-set stability, not
accuracy improvement**. The single-seed "GA helps" results observed in the E6 sweep did
not survive replication across seeds (Section 16), but the GA-selected feature subset
itself — dominated by `time`, `ejection_fraction`, and `serum_creatinine` — was highly
stable across five independent runs and, notably, matches Chicco & Jurman (2020)'s
independently derived feature-importance ranking for the same dataset, a source the GA
process had no access to during its search. This is treated in this report as a genuine
positive finding in its own right, even though it is not the finding that a
predetermined, in-favour-of-the-proposed-method narrative would have preferred: the
project's governing principle explicitly prohibits forcing GA to appear to win, and the
evidence here supports "GA as a principled feature-reduction tool" rather than "GA as an
accuracy-improving tool" on this dataset.

Fourth, the `time` feature's outsized predictive contribution must be read as a
**modelling caveat rather than a clinical discovery**. In survival-style clinical data,
a short recorded follow-up interval correlates with the death event almost by
construction, since a patient who dies is not observed further. `time`'s consistent
selection by GA and its central role in the dataset's own origin literature both need to
be understood in this light, not presented as an independently surprising predictive
insight.

Fifth, the project's **binding constraint is sample size, not method choice**. Eighteen
of the twenty-seven statistical tests conducted were non-significant; given that the
complete-data baseline's own across-seed standard deviation in F1 (0.081) exceeds most
of the point-estimate gaps under test, the most parsimonious explanation for these
non-significant results is that a 299-row dataset and a 60-row held-out test set, even
sampled across 5 seeds, does not carry enough statistical power to resolve differences
this fine-grained — not that no real difference exists between the compared methods.

## 19. Existing-Work Comparison

Own results are compared against selected published literature contextually, without
presenting numbers computed on different datasets as a direct head-to-head result. This
project's RQ1 finding — that missingness rate matters more than mechanism — agrees with
Shadbahr et al. (2023). Its RQ2 finding — that no single imputer statistically dominates
— agrees with both Ren et al. (2024)'s systematic review of EHR imputation methods and
Aracri et al. (2025)'s comparison for dementia classification. Its RQ3 finding on
feature-selection stability agrees, independently, with Chicco & Jurman (2020)'s
feature-importance analysis for the same underlying dataset. Its RQ3 finding on accuracy
does **not** agree with Kumar & Sahoo (2017), who reported a favourable GA-plus-Naive-
Bayes result on a different cardiovascular dataset; a plausible, though unproven,
explanation offered here is that this project's small feature space (12 features) and
training set (239 rows) give a genetic wrapper search comparatively little room to find a
genuinely superior subset, and comparatively more room to overfit its own
cross-validation-based fitness signal — a known general risk of wrapper-based feature
selection on small, noisy datasets (Pudjihartono et al., 2022). This divergence is
reported directly, not omitted, consistent with the project's governing principle that
evidence should not be forced toward a predetermined conclusion.

## 20. Limitations

- The study uses a single primary dataset of 299 rows (239 train / 60 test), small by
  general machine-learning standards; this is an intentional scope-control decision
  restricting the main study to one primary dataset within a single-semester timeframe.
- Five seeds is a statistically thin repeated-measures design: 18 of 27 paired tests are
  non-significant, which should be read as "not distinguishable from noise at this
  sample size," not as "no real effect exists."
- No multiple-comparison correction was applied to the 27 statistical tests; at n = 5,
  any standard correction would make almost every comparison non-significant regardless
  of true effect size, so individual p-values are read as exploratory/suggestive rather
  than confirmatory.
- Each seed reseeds the entire pipeline, including the train/test split itself, not only
  model training — a more thorough but higher-variance multi-seed design than reseeding
  the model alone; this should be kept in mind when comparing this project's variance
  estimates to studies that reseed only the model.
- E6 (Imputation + GA + Random Forest) multi-seed coverage is a 3-cell/3-seed subset,
  not the full 9-cell/5-seed grid, for compute-budget reasons; sufficient to show that
  the single-seed GA pattern does not replicate, but not an exhaustive test of every
  cell.
- GA fitness evaluation used 50 trees against the final model's 200-tree configuration,
  a documented roughly five-times cost reduction for the search process only; final
  reported comparisons use 200 trees on both sides, so reported performance numbers are
  unaffected, though the GA search itself operated on a noisier fitness signal than the
  final evaluation.
- KNN's k = 5 and IterativeImputer's settings are scikit-learn defaults, not
  cross-validation-tuned; neither choice is unreasonable, but neither is optimized.
- Missingness and imputation are scoped to the seven continuous features; the five
  binary clinical flags are not masked, since mean/median imputation is not a meaningful
  reconstruction target for a binary indicator.
- The `time` column's leakage-adjacency, discussed in Section 18 (Discussion) and
  Section 7 (Dataset), is retained as a documented methodological decision rather than
  removed, and its outsized influence on Random Forest performance should not be read as
  a genuine predictive discovery.
- Two literature entries — Kumar & Sahoo (2017) and Yaqoob et al. (2025) — carry a
  verification caveat: their full text could not be retrieved due to publisher access
  restrictions, so bibliographic details and reported findings are drawn from indexed
  metadata rather than a direct primary-source read. One entry, Grzesiak et al. (2025),
  is an unreviewed arXiv preprint and is labelled accordingly.
- No single master `run_all.py` execution script exists; the project instead uses one
  independently re-runnable script per experiment group, a documented scope-control
  simplification.

## 21. Future Work

- **Additional seeds** beyond the current five, to resolve the several near-threshold
  p-values observed in the statistical tests and give RQ1 and RQ2 more statistical
  power.
- **Full 9-cell/5-seed E6 coverage**, extending the current 3-cell/3-seed replication
  subset, to confirm across every mechanism-by-level cell (not only the three sampled)
  that GA does not reliably improve accuracy once imputation is already in the pipeline.
- **A second dataset for external validation**, most plausibly Chronic Kidney Disease
  (ranked second among the candidates in `docs/dataset_selection.md`), whose genuine
  natural missingness would let RQ1 be tested against real rather than only synthetic
  missingness.
- **Cross-validation-tuned KNN k and IterativeImputer settings**, rather than
  scikit-learn defaults, to check whether the disconnect between reconstruction quality
  and downstream prediction quality (Section 20) persists under tuning.
- **A hybrid filter-then-wrapper feature-selection approach** (Pudjihartono et al.,
  2022), to test whether GA's plausible overfitting risk on a small cross-validation
  signal (offered here as the explanation for the project's mixed RQ3 result) is reduced
  by pre-filtering the candidate feature space before the genetic search begins.
- **An explicit sensitivity analysis on the `time` feature**, rerunning the project's
  headline comparisons with `time` excluded, to directly quantify how much of the
  reported Random Forest performance depends on this leakage-adjacent feature.
- **An optional Autoencoder-based imputer**, only if a future, substantially larger
  dataset or a specific reviewer request justifies the added complexity; not pursued
  here, since the classical and iterative imputers already implemented are sufficient to
  answer RQ1 and RQ2 at this dataset's scale.

## 22. Conclusion

This project set out to determine how missing data affects clinical prediction, which
imputation method performs best, and whether GA-based feature selection improves
Random Forest prediction, using a controlled, repeated-seed experimental design on the
Heart Failure Clinical Records dataset. The evidence supports one clear, statistically
defensible conclusion: missingness itself, independent of its mechanism or the method
used to address it, measurably reduces prediction performance. No imputation method and
no missingness mechanism could be statistically distinguished from its alternatives at
this dataset's sample size, so no ranking among mean, median, KNN, and IterativeImputer,
nor among MCAR, MAR, and MNAR, is claimed. GA-based feature selection reliably and
reproducibly identified a small, clinically plausible feature subset that independently
matches this dataset's own origin literature, but did not reliably improve prediction
accuracy once tested across multiple seeds rather than a single split — a genuine,
non-forced finding that GA's demonstrated value here is principled dimensionality
reduction rather than an accuracy gain. Consistent with the project's governing
principle of not forcing a predetermined conclusion, these results — including the
non-significant comparisons and GA's mixed accuracy result — are reported as the actual,
replicated evidence produced by this study, and are offered as a defensible, honestly
limited research result rather than a claim of a superior proposed method.

## References

Full bibliographic detail with DOIs is provided in `report_data/references.bib` and
`results/literature_matrix.csv`; the fifteen references cited above are, in the order
first cited: Rubin (1976); van Buuren & Groothuis-Oudshoorn (2011); Stekhoven & Buhlmann
(2012); Troyanskaya et al. (2001); Breiman (2001); Leardi, Boggia, & Terrile (1992);
Demsar (2006); Shadbahr et al. (2023); Ren et al. (2024); Aracri et al. (2025); Chicco &
Jurman (2020); Kumar & Sahoo (2017); Pudjihartono et al. (2022); Yaqoob et al. (2025);
Grzesiak et al. (2025).
