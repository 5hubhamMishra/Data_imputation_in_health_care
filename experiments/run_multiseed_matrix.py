"""Phase 19: full multi-seed coverage of the E1-E3 imputation x RF-prediction
matrix (master prompt section 29).

`run_repeated_seeds.py` established the seed-threading pattern for two cells
(E0, E1-mean@MCAR20). This extends it to every mechanism x missingness level
x imputer cell: 3 x 3 x 4 x 5 seeds = 180 runs. Each run independently masks
the train/test splits, fits the imputer on the training split only, and
evaluates RF the same way as `run_prediction.py` - the only difference is
looping over all 5 established seeds instead of a single default seed.
"""

import pandas as pd

from src.config import METRICS_DIR
from src.data_loader import load_raw_data
from src.imputation import apply_imputer, fit_imputer
from src.missingness import MISSINGNESS_LEVELS, mask_mar, mask_mcar, mask_mnar
from src.prediction import train_evaluate_rf
from src.preprocessing import train_test_split_stratified

SEEDS = [42, 123, 2026, 7, 99]
METHODS = ["mean", "median", "knn", "iterative"]
MECHANISMS = {"MCAR": mask_mcar, "MAR": mask_mar, "MNAR": mask_mnar}
METRIC_COLUMNS = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]


def run_matrix() -> pd.DataFrame:
    df = load_raw_data()
    rows = []
    for seed in SEEDS:
        X_train, X_test, y_train, y_test = train_test_split_stratified(df, seed=seed)
        for mechanism, mask_fn in MECHANISMS.items():
            for level in MISSINGNESS_LEVELS:
                pct = int(level * 100)
                masked_train, _, _ = mask_fn(X_train, level, seed=seed)
                masked_test, _, _ = mask_fn(X_test, level, seed=seed)
                for method in METHODS:
                    imputer = fit_imputer(method, masked_train, seed=seed)
                    imputed_train = apply_imputer(imputer, masked_train)
                    imputed_test = apply_imputer(imputer, masked_test)
                    metrics = train_evaluate_rf(imputed_train, y_train, imputed_test, y_test, seed=seed)
                    rows.append({"seed": seed, "mechanism": mechanism, "missing_pct": pct, "method": method, **metrics})

    out = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(METRICS_DIR / "e1_e2_e3_multiseed.csv", index=False)
    return out


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std/min/max/successful-run count per mechanism x level x method
    cell (master prompt section 32)."""
    grouped = df.groupby(["mechanism", "missing_pct", "method"])[METRIC_COLUMNS]
    summary = grouped.agg(["mean", "std", "min", "max", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(METRICS_DIR / "e1_e2_e3_multiseed_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    runs = run_matrix()
    print(f"Ran {len(runs)} total (mechanism x level x method x seed) combinations.")
    summary = aggregate(runs)
    print(summary[["mechanism", "missing_pct", "method", "test_f1_mean", "test_f1_std"]].to_string(index=False))
