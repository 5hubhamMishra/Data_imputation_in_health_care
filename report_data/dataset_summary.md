# Dataset Summary

**Heart Failure Clinical Records** (UCI ML Repository, id 519, CC BY 4.0).
299 rows, 12 features, binary target `DEATH_EVENT`. Fully observed: 0 missing
values, 0 duplicates. Class balance 203 (0) / 96 (1), ~68%/32%. SHA-256
checksum and full provenance in `data/raw/heart_failure_metadata.json`.

Selected from 5 shortlisted UCI candidates (Heart Failure, Chronic Kidney
Disease, Heart Disease/Cleveland, ILPD, Cervical Cancer) scored across 12
master-prompt criteria — full comparison in `results/dataset_candidate_comparison.csv`
(copied here as `tables/dataset_candidate_comparison.csv`).

**Why this dataset**: it is the only candidate with zero natural
missingness, which lets the complete-data RF baseline use all 299 rows and
gives every intentionally-masked MCAR/MAR/MNAR cell a known ground-truth
value — the cleanest possible setup for RQ1 and for MAE/RMSE reconstruction
scoring. It also has a single clear binary target (unlike Cervical Cancer's
four correlated targets) and an established feature-importance literature
(Chicco & Jurman 2020) to check GA results against.

**Accepted trade-off**: 12 features is smaller than CKD (24) or Cervical
Cancer (36), so GA feature-count reduction looks less dramatic in absolute
terms; judged acceptable since a clean experimental design outweighs a
larger but confounded feature space (master prompt section 48).

**Leakage-relevant column**: `time` (days of follow-up) correlates with the
death event almost by construction in survival-style data. Kept in the
dataset (removing it would be an undocumented methodological choice) but
its outsized influence must be flagged explicitly in the Results/Discussion
sections, not presented as a genuine predictive discovery.

Full rationale and rejected-candidate reasoning: `docs/dataset_selection.md`.
