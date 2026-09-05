# Supervisor Progress Summary

## Project

Healthcare Data Imputation Using Machine Learning with Genetic Algorithm-Based
Feature Selection.

## Completion status

The project is complete for the defined single-dataset scope. The repository
contains the executed experiment outputs, validated analysis, report-data
package, literature review, and final academic report draft.

## Work completed

- Selected and integrity-checked the UCI Heart Failure Clinical Records dataset
  (299 rows, 12 features, binary `DEATH_EVENT`, no natural missing values).
- Implemented leakage-safe splitting, MCAR/MAR/MNAR masking at 10%, 20%, and
  30%, and ground-truth-preserving reconstruction scoring.
- Implemented mean, median, KNN, and IterativeImputer methods.
- Completed the 180-run multi-seed prediction matrix.
- Implemented GA feature selection and completed the nine-cell imputation-plus-
  GA sweep.
- Completed 27 paired statistical comparisons and a three-cell multi-seed GA
  replication subset.
- Verified 15 literature entries, prepared report assets, and drafted the
  final academic report.

## Defensible findings

Missingness measurably reduced prediction performance in 4 of 9 complete-data
comparisons (paired t-test, p < 0.05). No imputer or missingness mechanism was
statistically distinguishable from its alternatives at five seeds. GA produced
a stable feature-reduced subset, consistently selecting `time` and
`ejection_fraction`, but did not reliably improve prediction after replication.
The `time` feature remains a documented leakage-adjacent modelling caveat.

## Remaining limitations

The primary dataset is small, only five seeds were used, and the E6
imputation-plus-GA replication covers three of nine cells across three seeds.
These are reported limitations, not hidden gaps. Full details and future work
are in `docs/final_report.md`.

## Deliverables

- Final report: `docs/final_report.md`
- Report assets: `report_data/`
- Reproducibility audit: `docs/reproducibility_audit.md`
- Source pipeline: `src/`
- Experiment runners: `experiments/`
- Tests: `tests/`
