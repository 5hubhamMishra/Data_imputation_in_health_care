# Experiment Log

## 2026-09-05 — Repository scaffold

Commit: ea34f27

## 2026-09-05 — Dataset discovery and selection

Selected Heart Failure Clinical Records (UCI id 519). See
`docs/dataset_selection.md`. Commit: b47a3f1

## 2026-09-05 — Acquisition, preprocessing, EDA, E0 baseline

- Downloaded dataset from UCI static export, verified shape (299, 13),
  0 missing values, 0 duplicates, class balance 203/96 — matches the
  discovery-phase claim exactly. SHA-256: see
  `data/raw/heart_failure_metadata.json`.
- Added `src/config.py`, `src/data_loader.py`, `src/preprocessing.py`.
- Added `tests/test_data_loader.py` (3 tests, all passing).
- Generated EDA figures: class distribution, feature distributions,
  correlation matrix.
- Ran E0 complete-data RF baseline (seed 42): Accuracy 0.817, F1 0.667,
  ROC-AUC 0.883. Result: `results/metrics/e0_complete_rf_baseline.json`.
