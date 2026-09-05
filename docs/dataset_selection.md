# Dataset Selection

## Candidates Considered

Five public healthcare datasets were shortlisted from the UCI Machine Learning
Repository (all CC BY 4.0 licensed, all verified directly against their UCI
dataset pages on the access date below). Full comparison data is in
`results/dataset_candidate_comparison.csv`.

| Rank | Dataset | Rows | Features | Target | Natural Missingness |
|---|---|---|---|---|---|
| 1 | Heart Failure Clinical Records | 299 | 12 | DEATH_EVENT (binary) | None |
| 2 | Chronic Kidney Disease | 400 | 24 | ckd/notckd (binary) | Yes, extensive |
| 3 | Heart Disease (Cleveland) | 303 | 13 | num (0-4, presence/absence) | Minimal (a few cells) |
| 4 | ILPD (Indian Liver Patient) | 583 | 10 | Selector (binary) | None |
| 5 | Cervical Cancer (Risk Factors) | 858 | 36 | 4 boolean targets | Yes, substantial |

Access date: 2026-09-05. Sources:
- https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records
- https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
- https://archive.ics.uci.edu/dataset/45/heart+disease
- https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset
- https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors+data+set

## Ranking Methodology

Each candidate was scored 1-5 on twelve criteria from the master prompt
(healthcare relevance, missing-data suitability, sample size, feature
diversity, target clarity, class balance, reproducibility, public
availability, computational feasibility, privacy/licensing, literature
availability, GA suitability), summed to a score out of 60. Scores are in the
CSV. The scoring is a structured judgment aid, not a precise measurement —
the qualitative reasoning below is what actually drove selection.

The deciding factor was **missing-data suitability interacting with the
project's own phase order**: the master prompt requires a complete-data
Random Forest baseline (Phase 7) *before* the missingness framework is
introduced (Phase 8+), and requires that intentionally-masked cells have known
ground truth so imputation error is measurable (section 19). A dataset that
is already fully observed satisfies both requirements with no extra
machinery; a dataset with real natural missingness (CKD, Cervical Cancer)
would force a smaller complete-case subset for the baseline and extra
bookkeeping to keep natural and synthetic missingness separate throughout
every downstream experiment.

## Selected Dataset: Heart Failure Clinical Records

**Rationale:**

- **Fully observed**: every cell is real and complete, so the complete-data
  RF baseline uses the entire 299-row dataset with no complete-case
  reduction, and every masked cell for the MCAR/MAR/MNAR experiments has a
  known true value — the cleanest possible setup for RQ1 and imputation-error
  measurement (MAE/RMSE).
- **Single, clear binary target** (DEATH_EVENT), avoiding the
  multi-target ambiguity of Cervical Cancer.
- **Established GA/feature-selection literature**: this exact dataset
  (Chicco & Jurman 2020 and multiple follow-up feature-selection papers) is
  widely used for downstream RF prediction and feature importance/selection
  studies, giving a credible literature base for RQ4.
- **Computationally light** (299 rows, 12 features): the full experiment
  matrix (mechanism x missingness% x imputer x GA state x seed) can run
  repeatedly on an ordinary workstation, which the master prompt requires
  (section 40).
- **Meaningful class imbalance** (~32% positive) without being so skewed
  that standard classification metrics become uninformative.

**Trade-off accepted:** 12 features is smaller than CKD (24) or Cervical
Cancer (36), so GA-driven feature-count reduction will look less dramatic in
absolute terms. This is judged acceptable because the master prompt's
success criteria (section 48) are about a defensible research result, not
maximizing any single number, and a clean experimental design outweighs a
larger but confounded feature space.

## Why the Others Were Not Selected

- **Chronic Kidney Disease** (rank 2) is the strongest alternative and the
  natural candidate for an optional second dataset if external validation is
  pursued after the primary study (per section 2), specifically because its
  real natural missingness is directly relevant to RQ1 in a way Heart
  Failure cannot demonstrate. It was not selected as primary because its
  natural missingness complicates the complete-data baseline and requires
  extra care to avoid ever scoring natural missingness as imputation error.
- **Heart Disease (Cleveland)** (rank 3) is the most heavily cited dataset
  in this field, but its near-complete data (only a few missing cells) and
  smaller feature count (13) make it a weaker vehicle for both the
  missingness study and GA feature selection than Heart Failure.
- **ILPD** (rank 4) is complete and reasonably sized, but only 10 features
  limits GA feature-selection headroom, its class imbalance (~71/29) is more
  severe, and it has less directly-relevant literature for an
  imputation+GA pipeline.
- **Cervical Cancer (Risk Factors)** (rank 5) has the most rows and features
  on paper, but was dropped as primary because of three compounding issues:
  four correlated boolean targets requiring an arbitrary choice of one,
  severe class imbalance on every target, and heavy nonresponse-driven
  missingness (privacy-sensitive questions left blank) that is itself likely
  MNAR — this shrinks the usable complete-case ground truth and makes
  controlled masking experiments harder to interpret cleanly against
  already-substantial genuine missingness.

## Freeze

Per section 13 of the master prompt, the primary dataset is now frozen as
**Heart Failure Clinical Records** for the main study. No further
dataset-shopping will occur; a second dataset (most likely Chronic Kidney
Disease) may be considered only after the primary study is complete and only
if it materially strengthens external validation.
