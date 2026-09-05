# Results Summary

Consolidated results reference. All numbers here are drawn directly from
`docs/final_report.md` (Section 16, Results) and `report_data/results_summary.md`;
no new value or claim is introduced in this document. Full discussion is in
`docs/final_report.md` Sections 16-18.

## Complete-Data Baseline (E0)

Across 5 seeds (`results/metrics/repeated_seeds_e0.csv`): Accuracy
0.863 ± 0.040, Precision 0.802 ± 0.079, Recall 0.768 ± 0.156, F1
0.776 ± 0.081, ROC-AUC 0.918 ± 0.031. The single-seed (seed 42) result was
F1 0.667, ROC-AUC 0.883 — the gap between the single-seed number and the
5-seed mean is why no single-seed result is treated as final anywhere in
this project.

## Imputation Reconstruction Quality

Mean, median, and IterativeImputer performed similarly at every missingness
level (`results/metrics/e1_mean_median_imputation.csv`,
`e3_iterative_imputation.csv`). KNN (k = 5, untuned) was consistently the
worst reconstructor by MAE/RMSE (`results/metrics/e2_knn_imputation.csv`).

## Prediction After Imputation (180-Run Multi-Seed Matrix)

F1 ranged approximately 0.615-0.742 across the nine mechanism × level cells
(`results/metrics/e1_e2_e3_multiseed_summary.csv`), degrading as missingness
rose from 10% to 30%. Reconstruction quality and downstream prediction
quality did not track one another: KNN, the worst reconstructor, was
sometimes competitive with or better than the other imputers on downstream
F1.

## Statistical Tests

27 paired comparisons (paired t-test with Wilcoxon cross-check,
`results/tables/statistical_tests.csv`): 9 imputer-vs-imputer, 9
mechanism-vs-mechanism, 9 complete-vs-imputed. Results:

| Comparison family | Significant (p<0.05) |
|---|---|
| Complete data vs. imputed data | 4 of 9 (MCAR 10%, MCAR 20%, MNAR 20%, MNAR 30%) |
| Imputer vs. imputer | 0 of 9 |
| Mechanism vs. mechanism | 0 of 9 |

No multiple-comparison correction was applied (documented choice: at n = 5
seeds, any standard correction would push the significance threshold low
enough that almost no comparison could ever reach it).

## GA Feature Selection (E5, Complete Data)

Across 5 independent GA runs, each converging to a 4-7 feature subset out of
12 (`results/tables/ga_feature_frequency.csv`):

| Feature | Selection frequency |
|---|---|
| time | 1.0 |
| ejection_fraction | 1.0 |
| serum_creatinine | 0.8 |
| diabetes | 0.6 |
| serum_sodium, platelets, anaemia | 0.4 |
| creatinine_phosphokinase, sex, high_blood_pressure, smoking | 0.2 |
| age | 0.0 |

## All-Features vs. GA-Selected (Complete Data, Seed 42)

Majority-vote subset (≥50% frequency: time, ejection_fraction,
serum_creatinine, diabetes — 4 of 12, 66.7% reduction),
`results/metrics/all_features_vs_ga_complete.csv`:

| Variant | n_features | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| all_features | 12 | 0.817 | 0.667 | 0.883 |
| ga_selected | 4 | 0.833 | 0.706 | 0.815 |

Mixed single-seed result: Accuracy/F1 improved, ROC-AUC fell.

## Imputation + GA + Random Forest (E6)

Across the 9 mechanism × level cells at seed 42
(`results/metrics/e6_imputed_ga_rf.csv`): GA improved F1 in 3 of 9 cells,
matched it in 1 of 9, reduced it in 5 of 9. The three most informative cells
(largest apparent GA gain, largest apparent GA loss, a near-tie) were
replicated across 3 seeds (`results/metrics/e6_multiseed_subset.csv`).
Averaged over those 3 seeds, GA F1 was at or below all-features F1 in every
cell:

| Cell | All-features F1 (3-seed mean) | GA F1 (3-seed mean) |
|---|---|---|
| MCAR 30% | 0.647 | 0.632 |
| MAR 20% | 0.724 | 0.676 |
| MNAR 30% | 0.634 | 0.607 |

The single-seed "GA gain" at MCAR 30% did not replicate.

## Headline Findings

1. Missingness itself measurably reduces prediction performance relative to
   complete data (the one statistically majority-significant test family).
2. No imputation method or missingness mechanism is statistically
   distinguishable from its alternatives at this sample size.
3. GA reliably selects a stable, clinically plausible feature subset
   (matching the dataset's own origin-paper feature-importance ranking) but
   does not reliably improve prediction accuracy once tested across
   multiple seeds.

See `docs/research_question_analysis.md` for the RQ1-RQ4 analysis and
`docs/final_report.md` Section 18 for full discussion.
