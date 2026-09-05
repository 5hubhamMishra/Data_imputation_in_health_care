"""Controlled missingness injection for imputation experiments.

The primary dataset is naturally fully complete, so every masked cell here
is synthetic and its ground truth is always known — imputation error is
computed exactly on the intentionally hidden cells later (master prompt
section 19). Masking must be applied to the training split only, never
before the train/test split (section 17's leakage-prevention flow).

MCAR, MAR, and MNAR are all implemented. MAR/MNAR only cover 3 of the 7
continuous features each (master prompt section 20 asks for a documented
conditioning scheme per affected feature, not blanket coverage) — the same
3 target columns are reused across both mechanisms so their effect on
reconstruction/prediction is directly comparable to each other and to MCAR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONTINUOUS_FEATURES, PROTECTED_COLUMNS

MISSINGNESS_LEVELS = (0.10, 0.20, 0.30)

# MAR: missingness probability in the key column depends on another
# *observed* column's value (never the key column's own value), with a
# documented clinical rationale (master prompt section 20).
#   serum_creatinine  ~ age (high):      older patients more likely to have
#                       an incomplete/delayed renal panel recorded.
#   ejection_fraction ~ time (low):      a short follow-up/observation
#                       window means the echocardiogram is less likely to
#                       have been completed and recorded.
#   serum_sodium      ~ creatinine_phosphokinase (high): a high CPK signals
#                       an acute cardiac/muscle injury (urgent presentation)
#                       where a full electrolyte panel is more likely to be
#                       deprioritized or missing.
MAR_CONDITIONING = {
    "serum_creatinine": ("age", "high"),
    "ejection_fraction": ("time", "low"),
    "serum_sodium": ("creatinine_phosphokinase", "high"),
}

# MNAR: missingness probability in a column depends on that column's own
# (masked) value — a defensible proxy for "more severe/unstable readings are
# less likely to be fully charted" (master prompt section 20).
#   ejection_fraction low:  a critically low ejection fraction reflects a
#                     severe presentation where the echo workup is more
#                     often incomplete.
#   serum_creatinine high:  markedly elevated creatinine (acute kidney
#                     injury) is associated with repeat/redraw delays and
#                     reporting gaps in unstable patients.
#   serum_sodium low:       severe hyponatremia readings are more often
#                     flagged for a repeat test, with the original entry
#                     sometimes left unrecorded.
MNAR_CONDITIONING = {
    "ejection_fraction": "low",
    "serum_creatinine": "high",
    "serum_sodium": "low",
}


def _weighted_missing_indices(
    values: pd.Series, n_missing: int, rng: np.random.Generator, favor: str
) -> np.ndarray:
    """Sample `n_missing` indices without replacement, weighted by rank of
    `values` so that `favor='high'` gives higher missingness probability to
    higher values (and `favor='low'` to lower values) — a mild, monotonic
    probability gradient rather than a deterministic cutoff, so the
    mechanism stays probabilistic like real MAR/MNAR."""
    ranks = values.rank(method="first")  # 1..n, ascending
    weights = ranks if favor == "high" else (len(values) + 1 - ranks)
    weights = weights / weights.sum()
    return rng.choice(values.index.to_numpy(), size=n_missing, replace=False, p=weights.to_numpy())


def mask_mcar(
    df: pd.DataFrame,
    missing_frac: float,
    seed: int,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Mask `missing_frac` of cells independently at random (MCAR) in each
    eligible column of `df`.

    Returns:
        masked_df: copy of df with NaN at masked cells.
        ground_truth: same shape/columns as the masked subset, holding the
            original values only at masked cells (NaN elsewhere) — used to
            score imputation error on hidden cells only.
        actual_fractions: {column: realized missing fraction}, which can
            differ slightly from `missing_frac` due to rounding.
    """
    columns = [c for c in (columns or CONTINUOUS_FEATURES) if c not in PROTECTED_COLUMNS]
    rng = np.random.default_rng(seed)

    masked = df.copy()
    ground_truth = pd.DataFrame(np.nan, index=df.index, columns=columns)
    actual_fractions: dict[str, float] = {}

    for col in columns:
        n = len(df)
        n_missing = round(n * missing_frac)
        masked_idx = rng.choice(df.index.to_numpy(), size=n_missing, replace=False)
        ground_truth.loc[masked_idx, col] = df.loc[masked_idx, col]
        masked.loc[masked_idx, col] = np.nan
        actual_fractions[col] = n_missing / n

    return masked, ground_truth, actual_fractions


def mask_mar(
    df: pd.DataFrame,
    missing_frac: float,
    seed: int,
    conditioning: dict[str, tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Mask `missing_frac` of cells in each key column of `conditioning`,
    with probability weighted by another *observed* column's value (see
    MAR_CONDITIONING for the documented scheme). Same return shape as
    `mask_mcar`."""
    conditioning = conditioning or MAR_CONDITIONING
    columns = [c for c in conditioning if c not in PROTECTED_COLUMNS]
    rng = np.random.default_rng(seed)

    masked = df.copy()
    ground_truth = pd.DataFrame(np.nan, index=df.index, columns=columns)
    actual_fractions: dict[str, float] = {}

    for col in columns:
        cond_col, favor = conditioning[col]
        n = len(df)
        n_missing = round(n * missing_frac)
        masked_idx = _weighted_missing_indices(df[cond_col], n_missing, rng, favor)
        ground_truth.loc[masked_idx, col] = df.loc[masked_idx, col]
        masked.loc[masked_idx, col] = np.nan
        actual_fractions[col] = n_missing / n

    return masked, ground_truth, actual_fractions


def mask_mnar(
    df: pd.DataFrame,
    missing_frac: float,
    seed: int,
    conditioning: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Mask `missing_frac` of cells in each key column of `conditioning`,
    with probability weighted by the column's *own* value (see
    MNAR_CONDITIONING for the documented scheme). Same return shape as
    `mask_mcar`."""
    conditioning = conditioning or MNAR_CONDITIONING
    columns = [c for c in conditioning if c not in PROTECTED_COLUMNS]
    rng = np.random.default_rng(seed)

    masked = df.copy()
    ground_truth = pd.DataFrame(np.nan, index=df.index, columns=columns)
    actual_fractions: dict[str, float] = {}

    for col in columns:
        favor = conditioning[col]
        n = len(df)
        n_missing = round(n * missing_frac)
        masked_idx = _weighted_missing_indices(df[col], n_missing, rng, favor)
        ground_truth.loc[masked_idx, col] = df.loc[masked_idx, col]
        masked.loc[masked_idx, col] = np.nan
        actual_fractions[col] = n_missing / n

    return masked, ground_truth, actual_fractions


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.preprocessing import train_test_split_stratified

    df = load_raw_data()
    X_train, X_test, y_train, y_test = train_test_split_stratified(df)
    masked, gt, frac = mask_mcar(X_train, 0.20, seed=42)
    assert frac  # sanity: at least ran
    print("MCAR 20% actual fractions:", frac)
