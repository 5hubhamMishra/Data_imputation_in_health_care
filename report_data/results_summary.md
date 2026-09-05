# Results Summary

All numbers below are traceable to files in `results/` (copies of the
report-ready tables/figures are in `tables/` and `figures/` alongside this
file).

**E0 complete-data baseline**, 5-seed mean (`results/metrics/repeated_seeds_e0.csv`):
Accuracy 0.863±0.040, Precision 0.802±0.079, Recall 0.768±0.156, F1
0.776±0.081, ROC-AUC 0.918±0.031. Single-seed (seed 42) held-out numbers:
F1 0.667, ROC-AUC 0.883 — the spread between these two illustrates why
single-seed point estimates are not treated as final claims anywhere in
this project.

**Imputation reconstruction quality** (`results/metrics/e1_mean_median_imputation.csv`,
`e2_knn_imputation.csv`, `e3_iterative_imputation.csv`): mean, median, and
IterativeImputer perform similarly at every level; KNN (k=5, untuned) is
consistently the worst reconstructor by MAE/RMSE on this dataset's feature
scales.

**RF-after-imputation, multi-seed** (`results/metrics/e1_e2_e3_multiseed_summary.csv`,
180 runs): F1 ranges roughly 0.615–0.742 across the 9 mechanism×level cells
depending on imputer/level; degrades as missingness rises from 10% to 30%,
as expected. Reconstruction quality (MAE/RMSE) and downstream prediction
quality (F1/ROC-AUC) do not track each other — the worst reconstructor
(KNN) is sometimes competitive or best on downstream F1.

**Statistical tests** (`results/tables/statistical_tests.csv`, 27 paired
tests, full detail in `docs/statistical_analysis.md`): 4/9 complete-vs-imputed
comparisons significant at p<0.05 (MCAR 10%/20%, MNAR 20%/30%); 0/9
imputer-vs-imputer significant; 0/9 mechanism-vs-mechanism significant.

**GA feature selection** (`results/tables/ga_feature_frequency.csv`,
`figures/ga_feature_frequency.png`, `figures/ga_convergence.png`): `time`
and `ejection_fraction` selected in 5/5 independent runs, `serum_creatinine`
in 4/5, `age` in 0/5 — a stable, clinically plausible core subset.

**All-features vs GA** (`results/metrics/all_features_vs_ga_complete.csv`,
seed 42, complete data): 12→4 features (66.7% reduction); Accuracy
0.817→0.833, F1 0.667→0.706, ROC-AUC 0.883→0.815 (mixed — up on two
metrics, down on one).

**Imputation + GA + RF, 4-way comparison** (`results/tables/four_way_comparison.csv`,
`figures/four_way_comparison.png`): GA improved F1 in 3/9 E6 cells, tied in
1/9, hurt in 5/9 (single-seed). The 3-cell/3-seed replication
(`results/metrics/e6_multiseed_subset.csv`) shows GA at or below
all-features F1 in all 3 cells once averaged — the single-seed "wins" did
not replicate.

Full narrative and per-number provenance: `docs/progress_report.md`
("Actual Current Results", "GA Status" sections).
