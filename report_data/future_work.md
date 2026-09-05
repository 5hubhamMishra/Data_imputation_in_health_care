# Future Work

Evidence-motivated extensions, not commitments — each tied to a specific
limitation recorded in `limitations.md`.

- **More seeds** (beyond the current 5) to resolve the several near-threshold
  p-values (0.05–0.09) in the statistical tests and give RQ1/RQ2 more power.
- **Full 9-cell/5-seed E6 (Imputation+GA+RF) matrix**, extending the current
  3-cell/3-seed replication subset, to confirm the "GA does not reliably
  help once imputation is in the pipeline" finding holds across all
  mechanism×level cells, not just the 3 sampled.
- **Second dataset for external validation** (most likely Chronic Kidney
  Disease, ranked #2 in `docs/dataset_selection.md`), per master prompt
  section 2 — its real natural missingness would let RQ1 be tested against
  genuine rather than only synthetic missingness.
- **CV-tuned KNN k and IterativeImputer settings**, rather than scikit-learn
  defaults, to check whether the KNN-worst-reconstructor / KNN-competitive-
  predictor disconnect persists under tuning.
- **Hybrid filter-then-wrapper feature selection** (Pudjihartono et al.
  2022), to check whether GA's overfitting-to-small-CV-signal risk (the
  project's plausible explanation for its mixed RQ3 result) is reduced by
  pre-filtering the search space.
- **Explicit sensitivity analysis on the `time` column**: rerun the
  headline comparisons with `time` excluded, to quantify how much of RF's
  reported performance depends on this leakage-adjacent feature.
- **Optional Autoencoder imputer** (master prompt E4), only if a future
  larger dataset or a specific reviewer request justifies the added
  complexity — not pursued here since the classical/ML imputer set already
  answers RQ1/RQ2 on this dataset's scale.
