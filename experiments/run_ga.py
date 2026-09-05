"""Phases 14-16: GA feature selection on complete (unimputed) data (E5),
plus the all-features-vs-GA comparison (E5 vs E0, master prompt section 27).

Runs the GA once per seed (GA_SEEDS) on that seed's own train/test split, to
get a feature-selection-frequency picture across independent runs rather
than trusting a single search (master prompt section 26). The all-features
comparison reuses the seed-42 E0 baseline already on disk rather than
retraining it.
"""

import json

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, GA_SEEDS, METRICS_DIR, RANDOM_SEED, TABLES_DIR
from src.data_loader import load_raw_data
from src.genetic_selection import run_ga
from src.prediction import train_evaluate_rf
from src.preprocessing import train_test_split_stratified


def run_ga_across_seeds() -> list[dict]:
    df = load_raw_data()
    runs = []
    for seed in GA_SEEDS:
        X_train, X_test, y_train, y_test = train_test_split_stratified(df, seed=seed)
        result = run_ga(X_train, y_train, seed=seed)
        runs.append(result)
        print(f"seed {seed}: best_fitness={result['best_fitness']:.4f}, "
              f"n_selected={len(result['selected_features'])}, features={result['selected_features']}")
    return runs


def save_convergence_plot(run: dict, path):
    log = pd.DataFrame(run["generation_log"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(log["generation"], log["best_fitness"], label="Best fitness", linewidth=2)
    ax.plot(log["generation"], log["mean_fitness"], label="Mean fitness", linestyle="--")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (CV-F1 - feature-count penalty)")
    ax.set_title(f"GA Convergence (seed={run['seed']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_feature_frequency(runs: list[dict], all_features: list[str], csv_path, fig_path) -> pd.DataFrame:
    n_runs = len(runs)
    counts = {f: 0 for f in all_features}
    for run in runs:
        for f in run["selected_features"]:
            counts[f] += 1

    freq_df = pd.DataFrame({
        "feature": list(counts.keys()),
        "selection_count": list(counts.values()),
    })
    freq_df["selection_frequency"] = freq_df["selection_count"] / n_runs
    freq_df = freq_df.sort_values("selection_frequency", ascending=False).reset_index(drop=True)
    freq_df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(freq_df["feature"], freq_df["selection_frequency"])
    ax.set_xlabel(f"Selection frequency (across {n_runs} GA runs)")
    ax.set_title("GA Feature-Selection Frequency")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    return freq_df


def compare_all_features_vs_ga(freq_df: pd.DataFrame) -> pd.DataFrame:
    """All-features RF (E0, seed 42) vs RF restricted to the majority-vote
    GA feature subset (selected in >=3 of 5 runs), same seed-42 split."""
    with open(METRICS_DIR / "e0_complete_rf_baseline.json") as f:
        e0 = json.load(f)

    majority_features = freq_df[freq_df["selection_frequency"] >= 0.5]["feature"].tolist()

    df = load_raw_data()
    X_train, X_test, y_train, y_test = train_test_split_stratified(df, seed=RANDOM_SEED)
    ga_metrics = train_evaluate_rf(X_train[majority_features], y_train, X_test[majority_features], y_test)

    rows = [
        {
            "variant": "all_features",
            "n_features": e0["n_features"],
            "features": ",".join(X_train.columns),
            "test_accuracy": e0["test_accuracy"],
            "test_precision": e0["test_precision"],
            "test_recall": e0["test_recall"],
            "test_f1": e0["test_f1"],
            "test_roc_auc": e0["test_roc_auc"],
            "runtime_seconds": e0["runtime_seconds"],
        },
        {
            "variant": "ga_selected",
            "n_features": len(majority_features),
            "features": ",".join(majority_features),
            **{k: ga_metrics[k] for k in
               ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc", "runtime_seconds"]},
        },
    ]
    out = pd.DataFrame(rows)
    out["feature_reduction_pct"] = (1 - out["n_features"] / e0["n_features"]) * 100
    return out


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()
    X_train, _, _, _ = train_test_split_stratified(df, seed=RANDOM_SEED)
    all_features = list(X_train.columns)

    runs = run_ga_across_seeds()

    seed42_run = next(r for r in runs if r["seed"] == RANDOM_SEED)
    save_convergence_plot(seed42_run, FIGURES_DIR / "ga_convergence.png")

    freq_df = save_feature_frequency(
        runs, all_features,
        TABLES_DIR / "ga_feature_frequency.csv",
        FIGURES_DIR / "ga_feature_frequency.png",
    )
    print("\nFeature selection frequency:")
    print(freq_df.to_string(index=False))

    comparison = compare_all_features_vs_ga(freq_df)
    comparison.to_csv(METRICS_DIR / "all_features_vs_ga_complete.csv", index=False)
    print("\nAll-features vs GA-selected (complete data, seed 42):")
    print(comparison[["variant", "n_features", "feature_reduction_pct", "test_accuracy", "test_f1", "test_roc_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
