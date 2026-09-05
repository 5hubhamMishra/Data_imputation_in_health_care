# Statistical Analysis

Master prompt sections 29, 32, 33. Uses the full multi-seed matrix
(`results/metrics/e1_e2_e3_multiseed.csv`: 3 mechanisms x 3 missingness
levels x 4 imputers x 5 seeds = 180 runs) and the existing 5-seed E0
baseline (`results/metrics/repeated_seeds_e0.csv`).

## Method

Every comparison is a **paired t-test** (plus Wilcoxon signed-rank as a
non-parametric cross-check) across the 5 seeds {42, 123, 2026, 7, 99}.
Pairing is valid because `train_test_split_stratified(df, seed=seed)`
produces the *same held-out test set* for every mechanism/level/method run
at a given seed — the evaluation set is identical across the two arms of
each comparison, which is what pairing requires. The intentional-missingness
draws differ between mechanisms/methods, but that is the treatment being
compared, not a source of mismatched pairing.

**No multiple-comparison correction is applied**, by choice, not oversight:
with only 5 seeds, any standard correction (Bonferroni, Benjamini-Hochberg)
would push the family-wise significance threshold low enough that almost no
comparison could ever reach it, regardless of true effect size. These tests
are exploratory diagnostics for the results discussion — identifying which
patterns are worth discussing versus which are noise — not a confirmatory
hypothesis-testing family where a single false positive has an
external cost. Raw p-values are reported and interpreted with this
limitation stated explicitly (section 33 asks for correction only "where
genuinely necessary").

**Sample size caveat**: n=5 is thin. A paired t-test with 5 pairs has low
power to detect small-to-moderate effects; a non-significant result here is
evidence of "not distinguishable from noise at this sample size," not
evidence of "no real effect." All results below are read with that caveat.

Full results: `results/tables/statistical_tests.csv` (27 comparisons).

## Findings

### 1. Imputer vs imputer (9 tests, one per mechanism x level cell)

Best vs second-best imputer by mean F1, within each cell. **0/9 significant**
at p<0.05 (all p >= 0.42, most p > 0.7). The imputer rankings visible in the
single-seed results (e.g. KNN nearly matching the complete-data baseline at
10% MCAR) do not hold up as a real, seed-independent effect — with 5-seed
noise this large (F1 std 0.03-0.17 per cell), no imputer is reliably better
than any other on this dataset at this sample size.

### 2. Mechanism vs mechanism (9 tests, 3 pairs at each of 3 levels)

Each mechanism's own best-performing imputer at that level, compared
pairwise (MCAR vs MAR, MCAR vs MNAR, MAR vs MNAR). **0/9 significant**
(smallest p = 0.088, MCAR vs MNAR at 20%). The apparent MAR/MNAR-vs-MCAR gap
noted in earlier single-seed results (and investigated as a possible
column-coverage confound, then fixed) is, with multi-seed data, not
statistically distinguishable from noise at any level. This confirms the
progress-report's prior conclusion reached by inspection (E0's own 5-seed
std already exceeded the single-seed mechanism gap) with an actual test:
**RQ1's "which mechanism is easiest" cannot be answered from this dataset's
60-row test set with 5 seeds** — the honest answer is "no significant
difference detected."

### 3. Cost of missingness: complete data (E0) vs each mechanism's best
imputer (9 tests, 3 mechanisms at each of 3 levels)

This is the one family where a real, distinguishable signal appears:

| level | mechanism (imputer) | t p-value | Wilcoxon p-value |
|---|---|---|---|
| 10% | MCAR (knn) | **0.017** | 0.125 |
| 10% | MAR (median) | 0.219 | 0.313 |
| 10% | MNAR (knn) | 0.303 | 0.375 |
| 20% | MCAR (mean) | **0.012** | 0.063 |
| 20% | MAR (knn) | 0.065 | 0.125 |
| 20% | MNAR (median) | **0.008** | 0.063 |
| 30% | MCAR (knn) | 0.050 | 0.125 |
| 30% | MAR (knn) | 0.338 | 0.313 |
| 30% | MNAR (knn) | **0.024** | 0.063 |

**4/9 t-tests significant at p<0.05** (MCAR 10%/20%, MNAR 20%/30%), with two
more borderline (MAR 20% at p=0.065, MCAR 30% at p=0.050 — right at the
threshold). No cell trended toward missingness being *better* than complete
data; every comparison has complete-data F1 higher than the imputed mean.
The Wilcoxon p-values are uniformly larger (less powerful with n=5 and often
tied ranks), so this is best read as t-test-supported, Wilcoxon-consistent
directionally but not independently significant — a real but not
overwhelming signal given the sample size.

**Interpretation for RQ1**: missing data measurably hurts downstream
prediction relative to complete data, even after applying the
best-available imputer for that cell — this is the one claim in this
analysis with actual statistical support, not just a point-estimate
difference. The *degree* of harm does not clearly differ by mechanism or by
imputer choice at this sample size (see findings 1-2 above).

## Limitations

- 5 seeds is a small sample for paired testing; several near-threshold
  p-values (0.05-0.09) would likely resolve with more seeds but that is not
  attempted here (compute budget).
- The mechanism/imputer selected as "best" per cell is itself a single-seed
  choice under one selection seed elsewhere in the pipeline (E6); this
  analysis independently confirms rankings using the full 5-seed data, so it
  does not inherit that particular selection bias, but per-cell "best
  imputer" is still chosen by the same 5-seed data being tested (not an
  independent holdout) - a mild look-ahead that is standard in exploratory
  model comparison but worth naming.
- No correction for multiple comparisons (documented above) — read
  individual p-values as suggestive, not confirmatory.
