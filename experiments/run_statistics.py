"""Phases 22/33: paired statistical tests on the multi-seed matrix (master
prompt sections 32-33).

All comparisons pair by seed: `train_test_split_stratified(df, seed=seed)`
produces the identical held-out test set for every mechanism/level/method run
at that seed, so seed-matched F1 differences are a valid paired sample even
though the intentional-missingness draws differ between mechanisms - what
pairing requires is the same evaluation set, which this guarantees.

With only 5 seeds, these are exploratory diagnostics for the results
discussion, not confirmatory hypothesis tests: a sample of 5 is far below
what any multiple-comparison correction could meaningfully control without
making every test trivially non-significant, so no correction is applied
here (a documented choice, not an oversight - master prompt section 33 asks
for correction only "where genuinely necessary").
"""

import itertools

import pandas as pd
from scipy import stats

from src.config import METRICS_DIR, TABLES_DIR


def paired_test(a, b, label_a: str, label_b: str, comparison: str) -> dict:
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        # Wilcoxon is undefined when all paired differences are zero.
        w_stat, w_p = float("nan"), float("nan")
    return {
        "comparison": comparison,
        "a": label_a,
        "b": label_b,
        "mean_a": float(pd.Series(a).mean()),
        "mean_b": float(pd.Series(b).mean()),
        "n_pairs": len(a),
        "t_stat": float(t_stat),
        "t_pvalue": float(t_p),
        "wilcoxon_stat": None if w_stat != w_stat else float(w_stat),
        "wilcoxon_pvalue": None if w_p != w_p else float(w_p),
    }


def main() -> pd.DataFrame:
    multi = pd.read_csv(METRICS_DIR / "e1_e2_e3_multiseed.csv")
    e0 = pd.read_csv(METRICS_DIR / "repeated_seeds_e0.csv").sort_values("seed")
    levels = sorted(multi["missing_pct"].unique())
    results = []

    # 1. Best vs second-best imputer within each mechanism x level cell.
    for (mech, pct), group in multi.groupby(["mechanism", "missing_pct"]):
        means = group.groupby("method")["test_f1"].mean().sort_values(ascending=False)
        top2 = means.index[:2].tolist()
        a = group[group.method == top2[0]].sort_values("seed")["test_f1"].to_numpy()
        b = group[group.method == top2[1]].sort_values("seed")["test_f1"].to_numpy()
        results.append(paired_test(a, b, top2[0], top2[1], f"imputer_vs_imputer:{mech}_{pct}pct"))

    # Each mechanism's best-performing imputer at each level (reused below).
    best_per_mech_level = {}
    for (mech, pct), group in multi.groupby(["mechanism", "missing_pct"]):
        best_method = group.groupby("method")["test_f1"].mean().idxmax()
        vec = group[group.method == best_method].sort_values("seed")["test_f1"].to_numpy()
        best_per_mech_level[(mech, pct)] = (best_method, vec)

    # 2. Mechanism vs mechanism at matched levels, each using its own best imputer.
    for pct in levels:
        for m1, m2 in itertools.combinations(["MCAR", "MAR", "MNAR"], 2):
            method1, vec1 = best_per_mech_level[(m1, pct)]
            method2, vec2 = best_per_mech_level[(m2, pct)]
            results.append(paired_test(vec1, vec2, f"{m1}({method1})", f"{m2}({method2})", f"mechanism_vs_mechanism:{pct}pct"))

    # 3. Cost of missingness: complete-data E0 vs each mechanism's best imputer, per level.
    e0_vec = e0["test_f1"].to_numpy()
    for pct in levels:
        for mech in ["MCAR", "MAR", "MNAR"]:
            method, vec = best_per_mech_level[(mech, pct)]
            results.append(paired_test(e0_vec, vec, "E0_complete", f"{mech}({method})", f"missingness_cost:{pct}pct"))

    out = pd.DataFrame(results)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES_DIR / "statistical_tests.csv", index=False)
    return out


if __name__ == "__main__":
    out = main()
    print(out.to_string(index=False))
    n_sig = int((out["t_pvalue"] < 0.05).sum())
    print(f"\n{n_sig}/{len(out)} comparisons significant at p<0.05 (paired t-test, uncorrected, n=5 pairs each).")
