"""Phases 10-12, 17-18: mean/median (E1), KNN (E2), and IterativeImputer
(E3) imputation on the MCAR/MAR/MNAR-masked training data, scored against
preserved ground truth on intentionally hidden cells only.
"""

import pandas as pd

from src.config import METRICS_DIR, PROCESSED_DATA_DIR
from src.imputation import (
    impute_iterative, impute_knn, impute_mean, impute_median, score_imputation,
)
from src.missingness import MISSINGNESS_LEVELS

MECHANISMS = ["MCAR", "MAR", "MNAR"]

# (experiment group, output filename, {method_name: impute_fn})
EXPERIMENT_GROUPS = [
    ("E1", "e1_mean_median_imputation.csv", {"mean": impute_mean, "median": impute_median}),
    ("E2", "e2_knn_imputation.csv", {"knn": impute_knn}),
    ("E3", "e3_iterative_imputation.csv", {"iterative": impute_iterative}),
]


def main():
    masked_by_mechanism_level = {}
    ground_truth_by_mechanism_level = {}
    for mechanism in MECHANISMS:
        prefix = mechanism.lower()
        for level in MISSINGNESS_LEVELS:
            pct = int(level * 100)
            key = (mechanism, pct)
            masked_by_mechanism_level[key] = pd.read_csv(
                PROCESSED_DATA_DIR / f"{prefix}_{pct}_train_masked.csv", index_col=0
            )
            ground_truth_by_mechanism_level[key] = pd.read_csv(
                PROCESSED_DATA_DIR / f"{prefix}_{pct}_ground_truth.csv", index_col=0
            )

    for group_id, out_name, methods in EXPERIMENT_GROUPS:
        rows = []
        for mechanism in MECHANISMS:
            for level in MISSINGNESS_LEVELS:
                pct = int(level * 100)
                masked = masked_by_mechanism_level[(mechanism, pct)]
                ground_truth = ground_truth_by_mechanism_level[(mechanism, pct)]

                for method_name, impute_fn in methods.items():
                    imputed = impute_fn(masked)
                    scores = score_imputation(imputed, ground_truth)
                    for col, metrics in scores.items():
                        rows.append(
                            {
                                "mechanism": mechanism,
                                "missing_pct": pct,
                                "method": method_name,
                                "feature": col,
                                "mae": metrics["mae"],
                                "rmse": metrics["rmse"],
                            }
                        )

        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(METRICS_DIR / out_name, index=False)

        summary = df.groupby(["mechanism", "method", "missing_pct"])[["mae", "rmse"]].mean().round(3)
        print(f"\n{group_id} ({out_name}):")
        print(summary)

    print(f"\nSaved per-feature metrics to {METRICS_DIR}")


if __name__ == "__main__":
    main()
