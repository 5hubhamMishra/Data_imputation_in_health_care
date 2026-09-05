# Discussion Points

Talking points for the report's Discussion section, each traceable to an
existing result — not new analysis.

- **Missingness itself, not mechanism or imputer, is the dominant driver of
  performance loss.** This is the project's only statistically supported
  claim (RQ1) and independently corroborates Shadbahr et al. (2023) on
  different datasets — worth foregrounding as the project's most defensible
  finding.
- **Reconstruction quality and downstream prediction quality disagree.**
  KNN is the worst imputer by MAE/RMSE but sometimes matches or beats other
  imputers on downstream F1. Worth discussing why: RF is robust to
  moderate feature-level noise in ways that reconstruction-error metrics do
  not capture directly.
- **GA's real contribution here is feature-set stability, not accuracy.**
  The single-seed "GA helps" results in E5/E6 did not survive multi-seed
  replication, but the selected subset (`time`, `ejection_fraction`,
  `serum_creatinine`) is stable across 5 independent runs and matches an
  external, independent source (Chicco & Jurman 2020) that the GA had no
  way to have seen. This is a genuine positive finding even though it is
  not the finding the master prompt's Final Principle (section 49) would
  have preferred if the project were forcing a win.
- **`time` is doing a lot of the predictive work and is leakage-adjacent.**
  In survival-style clinical data, short follow-up time correlates with
  the death event almost by construction. It was kept (removing it would
  be an undocumented methodological choice) but its outsized RF importance
  should be discussed as a modeling caveat, not presented as a clinical
  discovery.
- **Sample size, not method choice, is this project's binding constraint.**
  Every "not statistically significant" finding (18 of 27 tests) is most
  parsimoniously explained by n=5 seeds and a 60-row test set, not by an
  absence of real effects — E0's own 5-seed F1 std (0.081) exceeds most of
  the point-estimate gaps being tested.
- **Divergence from Kumar & Sahoo (2017)'s favorable GA+RF result is worth
  naming directly**, with a plausible (not proven) explanation: this
  project's 12-feature space and 239-row training set give a genetic
  wrapper search little room to find a genuinely superior subset and more
  room to overfit its own CV-based fitness signal — consistent with general
  wrapper-method overfitting risk on small, noisy datasets.
