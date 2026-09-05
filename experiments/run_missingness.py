"""Phase 8-9: build the MCAR missingness framework at 10/20/30% on the
training split, save masked data + ground truth for later imputation
experiments, and generate the required missingness EDA figures.
"""

import json

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RANDOM_SEED, RESULTS_DIR, TABLES_DIR
from src.data_loader import load_raw_data
from src.missingness import MISSINGNESS_LEVELS, mask_mcar
from src.preprocessing import train_test_split_stratified
from src.visualization import plot_missing_per_feature, plot_missingness_heatmap


def main():
    df = load_raw_data()
    X_train, X_test, y_train, y_test = train_test_split_stratified(df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for level in MISSINGNESS_LEVELS:
        pct = int(level * 100)
        masked, ground_truth, actual = mask_mcar(X_train, level, seed=RANDOM_SEED)

        masked.to_csv(PROCESSED_DATA_DIR / f"mcar_{pct}_train_masked.csv", index=True)
        ground_truth.to_csv(PROCESSED_DATA_DIR / f"mcar_{pct}_ground_truth.csv", index=True)

        for col, frac in actual.items():
            summary_rows.append(
                {"mechanism": "MCAR", "requested_pct": pct, "column": col, "actual_pct": round(frac * 100, 2)}
            )

        if pct == 20:
            plot_missing_per_feature(
                masked,
                "missing_per_feature_mcar20.png",
                "Missing values per feature — MCAR 20% (training split)",
            )
            plot_missingness_heatmap(
                masked,
                "missingness_heatmap_mcar20.png",
                "Missingness pattern — MCAR 20% (training split)",
            )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES_DIR / "missingness_summary.csv", index=False)

    print(summary_df.to_string(index=False))
    print(f"\nSaved masked/ground-truth CSVs for {len(MISSINGNESS_LEVELS)} levels to {PROCESSED_DATA_DIR}")
    print(f"Saved summary table to {TABLES_DIR / 'missingness_summary.csv'}")


if __name__ == "__main__":
    main()
