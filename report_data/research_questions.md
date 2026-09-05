# Research Questions — Condensed Answers

Full evidence trail and per-number citations: `docs/research_question_analysis.md`.

**RQ1 — How does missing data affect prediction performance?**
Missingness measurably hurts prediction relative to complete data — the one
claim in this project with actual statistical support (4/9 complete-vs-
imputed paired t-tests significant, p<0.05; every one of the 9 cells trends
the same direction even where not individually significant). Which
*mechanism* hurts more cannot be answered at this sample size: 0/9
mechanism-vs-mechanism comparisons reached significance. Matches Shadbahr
et al. (2023)'s independent finding that missingness rate, not mechanism or
imputer, drives most of the downstream variance.

**RQ2 — Which imputation method performs best?**
No imputer is statistically distinguishable from any other (0/9
imputer-vs-imputer tests significant, all p≥0.42). Point-estimate rankings
exist (e.g. KNN edges out others at MCAR 10%) but are directional, not
conclusive, given per-cell F1 std of 0.03–0.17 across seeds. Matches Aracri
et al. (2025)'s finding on a comparable clinical task.

**RQ3 — Can GA improve prediction, or preserve it with fewer features?**
Mixed and largely non-replicating on the accuracy side: single-seed "wins"
in some E6 cells did not hold up when averaged across 3 seeds (GA at or
below all-features F1 in all 3 checked cells). What *does* hold up: GA
reliably and reproducibly selects a small, stable, clinically plausible
feature subset (`time`, `ejection_fraction` in 5/5 runs) that independently
matches the dataset's own origin paper's feature-importance finding (Chicco
& Jurman 2020). Conclusion: GA is a reliable feature-reduction tool here,
not a reliable accuracy-improvement tool.

**RQ4 — How does Imputation + GA + RF compare with baselines and literature?**
Own-results ordering (`results/tables/four_way_comparison.csv`): complete+RF
highest, imputed+RF lower (consistent with RQ1), and adding GA to either
does not produce a consistent further improvement. Literature comparison is
contextual only (different datasets, never presented head-to-head):
agrees with Shadbahr et al. (2023), Ren et al. (2024), Aracri et al. (2025),
and Chicco & Jurman (2020); diverges from Kumar & Sahoo (2017)'s GA+RF
cardiovascular result where GA reportedly helped — reported honestly as a
genuine, dataset/scale-dependent divergence, not suppressed.

**Cross-cutting design note**: all 5 seeds redraw the train/test split
itself (not just model randomness), so the reported multi-seed variance
reflects the full pipeline's sensitivity — a more honest but also
higher-variance design than reseeding the model alone.
