# Literature Review

15 references were identified and verified against primary or indexed
sources (see `results/literature_matrix.csv` for full bibliographic detail
and per-paper notes). Two entries carry an explicit verification caveat:
Kumar & Sahoo (2017) and the raw abstract text of Yaqoob et al. (2025) could
not be fetched directly (403 responses from the publisher); their
bibliographic details and findings below are taken from search-engine
indexed summaries of the primary source rather than a direct read, and this
is flagged in the matrix. One entry (Grzesiak et al. 2025) is an arXiv
preprint, not yet peer-reviewed. All other entries were confirmed by
directly fetching the publisher/PMC/JMLR page.

## 1. Missing-data mechanisms (MCAR/MAR/MNAR)

The MCAR/MAR/MNAR typology this project's `src/missingness.py` implements
traces to **Rubin (1976)**, which established when a missingness mechanism
can be ignored for likelihood-based inference. Rubin's framework is
theoretical; more recent work has focused on how the mechanism interacts
with imputation and downstream modeling in practice. **Shadbahr et al.
(2023)**, evaluating three simulated and three real-world clinical
datasets, found via ANOVA decomposition that the *missingness rate itself*
(particularly in test data) explains most of the variance in downstream
classifier performance, while the *choice of imputation method* explains
comparatively little. This is a close match to this project's own
statistically-tested finding (`docs/statistical_analysis.md`): 4 of 9
complete-vs-imputed comparisons reached significance (p<0.05), while 0 of 9
imputer-vs-imputer and 0 of 9 mechanism-vs-mechanism comparisons did at
n=5 seeds. The convergence with an independent study using different
datasets strengthens confidence that this is a real pattern rather than an
artifact of this project's specific dataset or seed choice — though it is
not proof, since the two studies used different data.

## 2. Imputation methods

**Troyanskaya et al. (2001)** introduced KNN-based imputation for
microarray data, and **Stekhoven & Buhlmann (2012)** introduced MissForest,
an iterative random-forest-based imputer for mixed-type data, generally
outperforming KNN and MICE-style approaches in their evaluation. **van
Buuren & Groothuis-Oudshoorn (2011)** describe MICE, still the most common
comparator in EHR missing-data studies per **Ren et al. (2024)**'s
systematic review of 46 studies (2010-2024). This project implements mean,
median, KNN, and scikit-learn's `IterativeImputer` (a single-imputation,
lighter-weight relative of MICE/MissForest) rather than full MICE or
MissForest, consistent with the master prompt's scope-control directive to
avoid open-ended method proliferation — a decision also supported by
**Grzesiak et al. (2025)**'s (preprint) argument that a small set of
well-established imputers is usually sufficient in practice.

On a clinical dataset directly comparable in spirit to this project's,
**Aracri et al. (2025)** compared mean/median/KNN/MICE/MissForest ahead of
dementia classification and found MICE most consistent but no imputer
uniformly best across classifiers — again matching this project's finding
that imputer choice was not statistically distinguishable at the sample
size tested.

## 3. Random Forest for healthcare prediction

**Breiman (2001)** is the foundational citation for Random Forest itself.
**Chicco & Jurman (2020)** is the origin paper for this project's own
dataset (Heart Failure Clinical Records) and found that serum creatinine
and ejection fraction alone predict survival comparably to the full
12-feature set — notably, this project's own GA feature-selection runs
independently selected `ejection_fraction` and `serum_creatinine` as the
two most consistently chosen features (5/5 and 4/5 seeds respectively),
which agrees with Chicco & Jurman's clinical-importance finding even though
this project's GA did not know that result in advance.

## 4. Genetic Algorithm feature selection

**Leardi (1992)** is an early, foundational application of genetic
algorithms to feature subset selection. **Pudjihartono et al. (2022)**'s
review situates GA as one heuristic wrapper-search strategy among several
(alongside filter and embedded methods), and recommends hybrid
filter-then-wrapper pipelines for large feature spaces — a scale
consideration less relevant to this project's 12-feature dataset than to
genome-scale problems, but a useful pointer for future work.

Two applied studies offer a useful contrast to this project's own result.
**Kumar & Sahoo (2017)** report GA+Random Forest achieving the highest
accuracy (93.2%, with 6 selected features) among several feature-selection
methods on a cardiovascular disease dataset — a case where GA reportedly
helped. **Yaqoob et al. (2025)** report a *different* metaheuristic (Seagull
Optimization, not GA) improving Random Forest breast-cancer classification;
it is included here as a related metaheuristic-plus-RF healthcare pipeline,
not as GA evidence, and should not be cited as a genetic-algorithm result.

**This project's own GA result is mixed, not a clean win**: on complete
data, GA improved F1 (0.667→0.706) but reduced ROC-AUC (0.883→0.815) at
seed 42; across the small 3-cell/3-seed E6 multi-seed check, GA's
apparent single-seed gains did not survive averaging (mean GA F1 fell
slightly below mean all-features F1 in all 3 sampled cells). This
divergence from the Kumar & Sahoo cardiovascular case is worth discussing
rather than explaining away — a plausible factor is this project's very
small feature space (12 features) and modest sample size (239 training
rows), where a wrapper search has limited room to find a genuinely superior
subset and more room to overfit the CV-based fitness signal, consistent
with the general overfitting risk noted for wrapper methods on small,
noisy datasets in the feature-selection literature. This project makes no
claim that GA "does not work" in general — only that on this specific
dataset, at this scale, and across the seeds tested here, it did not
reliably outperform using all features. This should be stated plainly in
the final report per the master prompt's anti-cherry-picking principle
(section 49): do not force the proposed method to win.

## 5. Statistical methodology

**Demsar (2006)** recommends paired non-parametric tests (Wilcoxon
signed-rank / Friedman) over paired t-tests for classifier comparison,
noting that t-tests across resampled folds can inflate Type I error. This
project's `docs/statistical_analysis.md` reports both a paired t-test and
Wilcoxon signed-rank test for each comparison and notes the small sample
size (n=5 seeds) as a real limitation on statistical power — consistent
with Demsar's broader caution about drawing strong conclusions from limited
repeated measurements.

## Where this project agrees and disagrees with prior work

- **Agrees**: missingness itself is a bigger driver of prediction loss than
  imputer choice (Shadbahr et al. 2023); no single imputer dominates across
  contexts (Ren et al. 2024; Aracri et al. 2025); Random Forest's own
  feature importance on this exact dataset (Chicco & Jurman 2020) lines up
  with this project's independently-derived GA feature-selection frequency.
- **Diverges**: unlike Kumar & Sahoo (2017)'s GA+RF cardiovascular result,
  this project did not find a reliable GA improvement once tested across
  multiple seeds — reported honestly as a negative/mixed finding rather
  than suppressed or re-tuned to match the literature's more favorable
  pattern.
