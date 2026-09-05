# Reproducibility and Repository Audit

Date: 2026-09-05

## Repository state

- Branch: `main`
- Remote: `origin` → `https://github.com/5hubhamMishra/Data_imputation_in_health_care.git`
- Source, experiment, test, documentation, report-data, and result directories
  are present.
- The raw dataset is accompanied by provenance and SHA-256 metadata.
- `.gitignore` excludes caches, virtual environments, notebook checkpoints,
  and local secrets.

## Coverage checks

- E1-E3 multi-seed matrix: 180/180 runs present.
- Paired statistical analysis: 27/27 comparisons present.
- Complete-data GA: 5 seeds present.
- E6 imputation + GA + RF: 9/9 single-seed cells present; 3/9 cells have
  three-seed replication.
- Literature review: 15 entries present, with access caveats labelled.
- Final report and report-data package: present.

## Validation evidence

The final experiment audit recorded 24/24 tests passing, reproduced the seed-42
E0 metrics, confirmed train-only fitting for imputers and GA fitness, checked
referenced result files, and verified that the four significant missingness-
cost p-values match the report. Rerun again for this closing audit
(`python -m pytest -q`): 24/24 passing, unchanged. A scan of `src/`,
`experiments/`, and `docs/` for hardcoded local machine paths and of
`requirements.txt` against every third-party import used in the codebase
(`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `pytest`)
found no absolute paths and no missing dependency.

## Intentional scope decisions

No `run_all.py` wrapper or autoencoder was added. Separate experiment runners
are already independently reproducible, and an autoencoder is not justified
for 299 rows and 12 features. Full 9-cell/5-seed E6 replication remains future
work because its compute cost is explicitly disclosed in the report.
