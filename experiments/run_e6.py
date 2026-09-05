"""Phase 16 / experiment group E6: Imputation + GA + Random Forest.

For each of the 9 (mechanism, missingness_%) cells, the imputer with the best
downstream RF F1 in `e1_e2_e3_rf_prediction.csv` is carried forward (picked
per-cell from that file, not assumed), the GA (already validated on complete
data in E5) is run on the imputed training data only, and the GA-selected
subset is compared against the same cell's imputed-all-features RF number
(reused from E1-E3, not rerun).
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, METRICS_DIR, RANDOM_SEED, TABLES_DIR
from src.data_loader import load_raw_data
from src.genetic_selection import run_ga
from src.imputation import apply_imputer, fit_imputer
from src.missingness import MISSINGNESS_LEVELS, mask_mar, mask_mcar, mask_mnar
from src.prediction import train_evaluate_rf
from src.preprocessing import train_test_split_stratified

MECHANISMS = {"MCAR": mask_mcar, "MAR": mask_mar, "MNAR": mask_mnar}


def pick_best_imputer_per_cell(e1_e2_e3_path) -> pd.DataFrame:
    """Per (mechanism, missing_pct), the method with the highest test_f1 in
    the existing E1-E3 results - not assumed, read from the actual file."""
    df = pd.read_csv(e1_e2_e3_path)
    best = df.loc[df.groupby(["mechanism", "missing_pct"])["test_f1"].idxmax()]
    return best.reset_index(drop=True)


def run_e6_cell(mechanism: str, level: float, method: str, X_train, X_test, y_train, y_test) -> dict:
    mask_fn = MECHANISMS[mechanism]
    masked_train, _, _ = mask_fn(X_train, level, seed=RANDOM_SEED)
    masked_test, _, _ = mask_fn(X_test, level, seed=RANDOM_SEED)

    imputer = fit_imputer(method, masked_train, seed=RANDOM_SEED)
    imputed_train = apply_imputer(imputer, masked_train)
    imputed_test = apply_imputer(imputer, masked_test)

    # GA fitness sees only imputed_train/y_train - imputed_test never passed in,
    # matching the leakage-safe contract already verified for run_ga (E5).
    # population/generations reduced from E5's 30/30 (measured ~204s/cell at
    # 30/30, x9 cells > 30min): this only affects search thoroughness for the
    # 9-cell E6 sweep, not E5's reported complete-data GA result.
    ga_result = run_ga(imputed_train, y_train, seed=RANDOM_SEED, population_size=20, generations=15)
    selected = ga_result["selected_features"]

    ga_metrics = train_evaluate_rf(imputed_train[selected], y_train, imputed_test[selected], y_test)

    return {
        "mechanism": mechanism,
        "missing_pct": int(level * 100),
        "imputer_used": method,
        "feature_count_before": imputed_train.shape[1],
        "feature_count_after": len(selected),
        "selected_features": ",".join(selected),
        "ga_test_accuracy": ga_metrics["test_accuracy"],
        "ga_test_precision": ga_metrics["test_precision"],
        "ga_test_recall": ga_metrics["test_recall"],
        "ga_test_f1": ga_metrics["test_f1"],
        "ga_test_roc_auc": ga_metrics["test_roc_auc"],
    }


def main():
    e1_e2_e3_path = METRICS_DIR / "e1_e2_e3_rf_prediction.csv"
    best_per_cell = pick_best_imputer_per_cell(e1_e2_e3_path)

    df = load_raw_data()
    X_train, X_test, y_train, y_test = train_test_split_stratified(df)

    rows = []
    for _, cell in best_per_cell.iterrows():
        mechanism, missing_pct, method = cell["mechanism"], cell["missing_pct"], cell["method"]
        level = missing_pct / 100
        print(f"Running E6 cell: {mechanism} {missing_pct}% ({method}) ...", flush=True)
        result = run_e6_cell(mechanism, level, method, X_train, X_test, y_train, y_test)
        result["imputed_all_features_f1"] = cell["test_f1"]
        result["imputed_all_features_roc_auc"] = cell["test_roc_auc"]
        rows.append(result)
        print(f"  -> GA selected {result['feature_count_after']} features, "
              f"F1 {result['ga_test_f1']:.3f} (all-features F1 was {cell['test_f1']:.3f})", flush=True)

    out_df = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "e6_imputed_ga_rf.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved E6 results to {out_path}")

    build_four_way_comparison(out_df)


def build_four_way_comparison(e6_df: pd.DataFrame):
    """Assembles the master prompt's required 4-way comparison (section 32):
    complete+RF (E0), complete+GA+RF (E5), imputed+RF (E1-E3, reused),
    imputed+GA+RF (E6, this module) - one row per E6 cell, with the two
    complete-data numbers repeated as a constant reference."""
    with open(METRICS_DIR / "e0_complete_rf_baseline.json") as f:
        e0 = json.load(f)
    e5 = pd.read_csv(METRICS_DIR / "all_features_vs_ga_complete.csv")
    e5_ga = e5[e5["variant"] == "ga_selected"].iloc[0]

    out = e6_df.copy()
    out["complete_rf_f1"] = e0["test_f1"]
    out["complete_rf_roc_auc"] = e0["test_roc_auc"]
    out["complete_ga_rf_f1"] = e5_ga["test_f1"]
    out["complete_ga_rf_roc_auc"] = e5_ga["test_roc_auc"]
    out = out.rename(columns={
        "imputed_all_features_f1": "imputed_rf_f1",
        "imputed_all_features_roc_auc": "imputed_rf_roc_auc",
        "ga_test_f1": "imputed_ga_rf_f1",
        "ga_test_roc_auc": "imputed_ga_rf_roc_auc",
    })
    cols = [
        "mechanism", "missing_pct", "imputer_used", "feature_count_after",
        "complete_rf_f1", "complete_rf_roc_auc",
        "complete_ga_rf_f1", "complete_ga_rf_roc_auc",
        "imputed_rf_f1", "imputed_rf_roc_auc",
        "imputed_ga_rf_f1", "imputed_ga_rf_roc_auc",
    ]
    out = out[cols]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES_DIR / "four_way_comparison.csv", index=False)
    print("\nFour-way comparison (F1):")
    print(out[["mechanism", "missing_pct", "imputer_used", "complete_rf_f1",
               "complete_ga_rf_f1", "imputed_rf_f1", "imputed_ga_rf_f1"]].to_string(index=False))

    plot_four_way(out)


def plot_four_way(out: pd.DataFrame):
    labels = [f"{r.mechanism}\n{r.missing_pct}%" for r in out.itertuples()]
    x = range(len(out))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], out["imputed_rf_f1"], width, label="Imputed + all features")
    ax.bar([i + width / 2 for i in x], out["imputed_ga_rf_f1"], width, label="Imputed + GA")
    ax.axhline(out["complete_rf_f1"].iloc[0], color="black", linestyle="--", linewidth=1,
               label="Complete + all features (E0)")
    ax.axhline(out["complete_ga_rf_f1"].iloc[0], color="gray", linestyle=":", linewidth=1,
               label="Complete + GA (E5)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test F1")
    ax.set_title("Four-Way Comparison: Complete/Imputed x All-Features/GA")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "four_way_comparison.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
