"""Small multi-seed expansion of E6 (Imputation + GA + RF) for 3 cells chosen
from the seed-42 E6 result (`e6_imputed_ga_rf.csv`): the biggest GA gain
(MCAR 30%), the biggest GA loss (MNAR 30%), and a roughly-tied cell
(MAR 20%). Full 9-cell x 5-seed coverage (45 GA runs) is too expensive for
one cycle; this establishes whether the single-seed E6 pattern (GA helps in
some cells, hurts in others, no consistent direction) survives multiple
seeds for at least these three representative cells.

The "imputed + all features" side is read from the already-computed
`e1_e2_e3_multiseed.csv` (same seed, mechanism, level, method) rather than
rerun, since only the GA side is new here.
"""

from __future__ import annotations

import pandas as pd

from src.config import METRICS_DIR
from src.data_loader import load_raw_data
from src.genetic_selection import run_ga
from src.imputation import apply_imputer, fit_imputer
from src.missingness import mask_mar, mask_mcar, mask_mnar
from src.prediction import train_evaluate_rf
from src.preprocessing import train_test_split_stratified

SEEDS = [42, 123, 2026]
MECHANISMS = {"MCAR": mask_mcar, "MAR": mask_mar, "MNAR": mask_mnar}

# (mechanism, missing_pct, imputer) - imputer matches the one E6 already used
# for this cell (from e6_imputed_ga_rf.csv), so this is a genuine multi-seed
# repeat of that cell, not a different pipeline choice.
CELLS = [
    ("MCAR", 30, "iterative"),  # biggest GA gain at seed 42: F1 0.444 -> 0.500
    ("MNAR", 30, "iterative"),  # biggest GA loss at seed 42: F1 0.571 -> 0.452
    ("MAR", 20, "knn"),         # roughly tied at seed 42: F1 0.647 -> 0.647
]


def main() -> pd.DataFrame:
    df = load_raw_data()
    multiseed = pd.read_csv(METRICS_DIR / "e1_e2_e3_multiseed.csv")
    rows = []

    for mechanism, pct, method in CELLS:
        level = pct / 100
        mask_fn = MECHANISMS[mechanism]
        for seed in SEEDS:
            X_train, X_test, y_train, y_test = train_test_split_stratified(df, seed=seed)
            masked_train, _, _ = mask_fn(X_train, level, seed=seed)
            masked_test, _, _ = mask_fn(X_test, level, seed=seed)

            imputer = fit_imputer(method, masked_train, seed=seed)
            imputed_train = apply_imputer(imputer, masked_train)
            imputed_test = apply_imputer(imputer, masked_test)

            ga_result = run_ga(imputed_train, y_train, seed=seed, population_size=20, generations=15)
            selected = ga_result["selected_features"]
            ga_metrics = train_evaluate_rf(imputed_train[selected], y_train, imputed_test[selected], y_test, seed=seed)

            all_features_f1 = multiseed.loc[
                (multiseed.mechanism == mechanism) & (multiseed.missing_pct == pct)
                & (multiseed.method == method) & (multiseed.seed == seed),
                "test_f1",
            ].iloc[0]

            rows.append({
                "mechanism": mechanism, "missing_pct": pct, "imputer_used": method, "seed": seed,
                "feature_count_after": len(selected),
                "imputed_all_features_f1": float(all_features_f1),
                "imputed_ga_f1": ga_metrics["test_f1"],
                "imputed_ga_roc_auc": ga_metrics["test_roc_auc"],
            })
            print(f"{mechanism} {pct}% ({method}) seed={seed}: all-features F1={all_features_f1:.3f}, "
                  f"GA F1={ga_metrics['test_f1']:.3f} ({len(selected)} features)", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(METRICS_DIR / "e6_multiseed_subset.csv", index=False)
    return out


if __name__ == "__main__":
    out = main()
    print("\nPer-cell mean (3 seeds):")
    print(out.groupby(["mechanism", "missing_pct"])[["imputed_all_features_f1", "imputed_ga_f1"]].mean().to_string())
