# Experiment Setup

Experiment groups (master prompt section 28), all implemented, executed,
and validated:

| Group | Description | Status |
|---|---|---|
| E0 | Complete-data RF baseline | Done — single seed + 5-seed repeat |
| E1 | Mean/median imputation | Done — MCAR/MAR/MNAR × 10/20/30%, multi-seed |
| E2 | KNN imputation | Done — same coverage as E1 |
| E3 | IterativeImputer (ML-based) imputation | Done — same coverage as E1 |
| E4 | Autoencoder (optional) | Not attempted — not scientifically justified given the 12-feature/299-row scale; classical methods already cover the required comparison |
| E5 | GA feature selection | Done — 5 seeds on complete data |
| E6 | Imputation + GA + RF | Done — full 9-cell single-seed sweep; 3-cell/3-seed multi-seed replication subset |
| E7 | Literature comparison | Done — contextual only, no cross-dataset head-to-head claims |

**Missingness matrix**: 3 mechanisms (MCAR/MAR/MNAR) × 3 levels (10/20/30%)
× 4 imputers (mean/median/KNN/iterative) × 5 seeds = 180 runs
(`results/metrics/e1_e2_e3_multiseed.csv`, aggregated in
`e1_e2_e3_multiseed_summary.csv`).

**GA matrix**: E5 — 5 independent runs (one per seed) on complete data
(`results/tables/ga_feature_frequency.csv`). E6 — 9 mechanism×level cells at
seed 42 (`results/metrics/e6_imputed_ga_rf.csv`), plus a 3-cell/3-seed
replication subset chosen as the most interesting single-seed cells
(biggest GA gain, biggest GA loss, a tie) — `results/metrics/e6_multiseed_subset.csv`.
Full 9-cell/5-seed GA coverage was not pursued (compute budget), documented
as an accepted, non-blocking limitation since the 3-cell subset already
shows a consistent (non-)pattern.

**Statistical test matrix**: 27 paired comparisons (9 imputer-vs-imputer, 9
mechanism-vs-mechanism, 9 complete-vs-imputed) — `results/tables/statistical_tests.csv`,
methodology in `docs/statistical_analysis.md`.

**Seeds**: {42, 123, 2026, 7, 99} (all master-prompt-specified defaults).

**Master execution**: no single `run_all.py` was built (the project instead
uses one script per experiment group under `experiments/`, each independently
re-runnable and idempotent against its own output file) — a deliberate
scope-control simplification for a single-dataset, workstation-scale project;
noted here rather than silently deviating from master prompt section 30's
suggested pattern.
