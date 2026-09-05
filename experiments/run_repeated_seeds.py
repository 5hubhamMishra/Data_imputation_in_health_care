"""Phase 19: repeated-seed aggregation (master prompt section 29).

A single seed gives one point estimate on a 60-row held-out test set, which
is noisy enough that point-estimate differences between mechanisms/methods
are not on their own evidence of a real effect. This establishes the
mean +/- std pattern for two cases: the E0 complete-data baseline, and E1
mean-imputation under MCAR 20%. Full multi-seed coverage of every
mechanism x level x imputer combination is a later cycle's job.
"""

import pandas as pd

from src.config import METRICS_DIR
from src.data_loader import load_raw_data
from src.imputation import fit_imputer, apply_imputer
from src.missingness import mask_mcar
from src.prediction import train_evaluate_rf
from src.preprocessing import train_test_split_stratified

SEEDS = [42, 123, 2026, 7, 99]
METRIC_COLUMNS = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]


def _aggregate(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    agg = df[METRIC_COLUMNS].agg(["mean", "std"]).T
    agg.columns = ["mean", "std"]
    return agg


def run_e0_repeated_seeds() -> pd.DataFrame:
    df = load_raw_data()
    rows = []
    for seed in SEEDS:
        X_train, X_test, y_train, y_test = train_test_split_stratified(df, seed=seed)
        metrics = train_evaluate_rf(X_train, y_train, X_test, y_test, seed=seed)
        rows.append({"seed": seed, **metrics})

    out = pd.DataFrame(rows)
    out.to_csv(METRICS_DIR / "repeated_seeds_e0.csv", index=False)
    return out


def run_e1_mcar20_repeated_seeds() -> pd.DataFrame:
    df = load_raw_data()
    rows = []
    for seed in SEEDS:
        X_train, X_test, y_train, y_test = train_test_split_stratified(df, seed=seed)
        # Independent draws on each split, same as run_prediction.py.
        masked_train, _, _ = mask_mcar(X_train, 0.20, seed=seed)
        masked_test, _, _ = mask_mcar(X_test, 0.20, seed=seed)

        imputer = fit_imputer("mean", masked_train, seed=seed)
        imputed_train = apply_imputer(imputer, masked_train)
        imputed_test = apply_imputer(imputer, masked_test)

        metrics = train_evaluate_rf(imputed_train, y_train, imputed_test, y_test, seed=seed)
        rows.append({"seed": seed, **metrics})

    out = pd.DataFrame(rows)
    out.to_csv(METRICS_DIR / "repeated_seeds_e1_mcar20.csv", index=False)
    return out


if __name__ == "__main__":
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    e0_runs = run_e0_repeated_seeds()
    print("E0 complete-data baseline, per-seed runs:")
    print(e0_runs[["seed", *METRIC_COLUMNS]].to_string(index=False))
    print("\nE0 aggregate (mean +/- std over 5 seeds):")
    print(_aggregate(e0_runs.to_dict("records")))

    e1_runs = run_e1_mcar20_repeated_seeds()
    print("\nE1 mean-imputation @ MCAR 20%, per-seed runs:")
    print(e1_runs[["seed", *METRIC_COLUMNS]].to_string(index=False))
    print("\nE1 @ MCAR20 aggregate (mean +/- std over 5 seeds):")
    print(_aggregate(e1_runs.to_dict("records")))
