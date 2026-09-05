"""Phase 10: mean/median imputation on the MCAR-masked training data,
scored against preserved ground truth on intentionally hidden cells only.
"""

import json

import pandas as pd

from src.config import METRICS_DIR, PROCESSED_DATA_DIR
from src.imputation import impute_mean, impute_median, score_imputation
from src.missingness import MISSINGNESS_LEVELS


def main():
    rows = []
    for level in MISSINGNESS_LEVELS:
        pct = int(level * 100)
        masked = pd.read_csv(PROCESSED_DATA_DIR / f"mcar_{pct}_train_masked.csv", index_col=0)
        ground_truth = pd.read_csv(PROCESSED_DATA_DIR / f"mcar_{pct}_ground_truth.csv", index_col=0)

        for method_name, impute_fn in [("mean", impute_mean), ("median", impute_median)]:
            imputed = impute_fn(masked)
            scores = score_imputation(imputed, ground_truth)
            for col, metrics in scores.items():
                rows.append(
                    {
                        "mechanism": "MCAR",
                        "missing_pct": pct,
                        "method": method_name,
                        "feature": col,
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                    }
                )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(METRICS_DIR / "e1_mean_median_imputation.csv", index=False)

    summary = df.groupby(["method", "missing_pct"])[["mae", "rmse"]].mean().round(3)
    print(summary)
    print(f"\nSaved per-feature metrics to {METRICS_DIR / 'e1_mean_median_imputation.csv'}")


if __name__ == "__main__":
    main()
