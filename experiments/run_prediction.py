"""Phases 13, 17-18: Random Forest prediction after imputation (E1-E3)
under MCAR/MAR/MNAR, compared against the complete-data baseline (E0).

Both the training and held-out test splits are masked (same mechanism/level,
independent random draws), but every imputer is fit on the training split
only and applied to the test split via `.transform` — matching E0's split
and RF config exactly so the comparison is fair (master prompt sections
17, 23, 28).
"""

import pandas as pd

from src.config import METRICS_DIR, RANDOM_SEED
from src.data_loader import load_raw_data
from src.imputation import apply_imputer, fit_imputer
from src.missingness import MISSINGNESS_LEVELS, mask_mar, mask_mcar, mask_mnar
from src.prediction import train_evaluate_rf
from src.preprocessing import train_test_split_stratified

METHODS = ["mean", "median", "knn", "iterative"]
MECHANISMS = {"MCAR": mask_mcar, "MAR": mask_mar, "MNAR": mask_mnar}


def main():
    df = load_raw_data()
    X_train, X_test, y_train, y_test = train_test_split_stratified(df)

    rows = []
    for mechanism, mask_fn in MECHANISMS.items():
        for level in MISSINGNESS_LEVELS:
            pct = int(level * 100)
            # Independent draws on each split (different index -> different
            # random cells even with the same seed value).
            masked_train, _, _ = mask_fn(X_train, level, seed=RANDOM_SEED)
            masked_test, _, _ = mask_fn(X_test, level, seed=RANDOM_SEED)

            for method in METHODS:
                imputer = fit_imputer(method, masked_train, seed=RANDOM_SEED)
                imputed_train = apply_imputer(imputer, masked_train)
                imputed_test = apply_imputer(imputer, masked_test)

                metrics = train_evaluate_rf(imputed_train, y_train, imputed_test, y_test)
                rows.append({"mechanism": mechanism, "missing_pct": pct, "method": method, **metrics})

    out_df = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "e1_e2_e3_rf_prediction.csv"
    out_df.to_csv(out_path, index=False)

    print(out_df[["mechanism", "missing_pct", "method", "test_accuracy", "test_f1", "test_roc_auc"]].to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
