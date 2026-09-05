# Limitations

- **Single primary dataset, 299 rows** (239 train / 60 test) — small by
  general ML standards; intentional per the master prompt's scope control
  (section 2), which restricts the main study to one primary dataset.
- **n=5 seeds is statistically thin.** 18 of 27 paired tests are
  non-significant, which is evidence of "not distinguishable from noise at
  this sample size," not evidence of "no real effect."
  `docs/statistical_analysis.md`.
- **No multiple-comparison correction applied** to the 27 statistical tests
  (documented rationale: at n=5, any standard correction would push the
  significance threshold low enough that almost nothing could ever reach
  it, regardless of true effect size). Individual p-values are read as
  exploratory/suggestive, not confirmatory.
- **Seeds reseed the entire pipeline, including the train/test split**, not
  just model training — a more thorough but higher-variance multi-seed
  design than reseeding the RF alone; readers comparing to studies that
  reseed only the model should account for this.
- **E6 (Imputation+GA+RF) multi-seed coverage is a 3-cell/3-seed subset**,
  not the full 9-cell/5-seed grid, due to compute budget. Sufficient to show
  the single-seed GA pattern does not replicate, but not exhaustive.
- **GA fitness evaluation used 50 trees vs. the final 200-tree RF
  configuration** (`src/config.py`), a documented ~5x cost reduction for the
  search process only — final reported comparisons use 200 trees on both
  sides, so reported performance numbers are unaffected, but the GA search
  itself saw a noisier fitness signal than the final evaluation.
- **KNN's k=5 and IterativeImputer's settings are scikit-learn defaults,
  not CV-tuned.** Neither choice is unreasonable, but neither is optimized.
- **Missingness/imputation is scoped to the 7 continuous features**; the 5
  binary clinical flags are not masked, since mean/median imputation is not
  a meaningful reconstruction target for a 0/1 flag (`src/config.py`).
- **`time` column leakage-adjacency**: kept in the dataset per an
  explicit, documented decision, but its outsized influence on RF
  performance needs explicit treatment as a limitation in the final
  discussion, not a genuine predictive discovery (`docs/dataset_selection.md`,
  `docs/progress_report.md`).
- **Two literature entries carry a verification caveat**: Kumar & Sahoo
  (2017) and Yaqoob et al. (2025)'s full text could not be fetched
  (403 responses); bibliographic details and findings are taken from
  indexed metadata, not a direct primary-source read. One entry (Grzesiak
  et al. 2025) is an unreviewed arXiv preprint. All flagged in
  `results/literature_matrix.csv` and `references.bib`.
- **No master `run_all.py`** — the project uses one script per experiment
  group instead, each independently re-runnable; a scope-control
  simplification, documented in `experiment_setup.md`.
