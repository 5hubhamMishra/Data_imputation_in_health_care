"""Phases 8-9, 17-18: build the MCAR/MAR/MNAR missingness framework at
10/20/30% on the training split, save masked data + ground truth for later
imputation experiments, and generate the required missingness EDA figures.
"""

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RANDOM_SEED, TABLES_DIR
from src.data_loader import load_raw_data
from src.missingness import MISSINGNESS_LEVELS, mask_mar, mask_mcar, mask_mnar
from src.preprocessing import train_test_split_stratified
from src.visualization import plot_missing_per_feature, plot_missingness_heatmap

MECHANISMS = {"MCAR": mask_mcar, "MAR": mask_mar, "MNAR": mask_mnar}


def main():
    df = load_raw_data()
    X_train, X_test, y_train, y_test = train_test_split_stratified(df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for mechanism, mask_fn in MECHANISMS.items():
        prefix = mechanism.lower()
        for level in MISSINGNESS_LEVELS:
            pct = int(level * 100)
            masked, ground_truth, actual = mask_fn(X_train, level, seed=RANDOM_SEED)

            masked.to_csv(PROCESSED_DATA_DIR / f"{prefix}_{pct}_train_masked.csv", index=True)
            ground_truth.to_csv(PROCESSED_DATA_DIR / f"{prefix}_{pct}_ground_truth.csv", index=True)

            for col, frac in actual.items():
                summary_rows.append(
                    {"mechanism": mechanism, "requested_pct": pct, "column": col, "actual_pct": round(frac * 100, 2)}
                )

            if pct == 20:
                plot_missing_per_feature(
                    masked,
                    f"missing_per_feature_{prefix}20.png",
                    f"Missing values per feature — {mechanism} 20% (training split)",
                )
                plot_missingness_heatmap(
                    masked,
                    f"missingness_heatmap_{prefix}20.png",
                    f"Missingness pattern — {mechanism} 20% (training split)",
                )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES_DIR / "missingness_summary.csv", index=False)

    print(summary_df.to_string(index=False))
    print(f"\nSaved masked/ground-truth CSVs for {len(MECHANISMS)} mechanisms x {len(MISSINGNESS_LEVELS)} levels to {PROCESSED_DATA_DIR}")
    print(f"Saved summary table to {TABLES_DIR / 'missingness_summary.csv'}")


if __name__ == "__main__":
    main()
