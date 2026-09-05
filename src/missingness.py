"""Controlled missingness injection for imputation experiments.

The primary dataset is naturally fully complete, so every masked cell here
is synthetic and its ground truth is always known — imputation error is
computed exactly on the intentionally hidden cells later (master prompt
section 19). Masking must be applied to the training split only, never
before the train/test split (section 17's leakage-prevention flow).

Only MCAR is implemented this cycle. MAR/MNAR are added later as sibling
`mask_mar` / `mask_mnar` functions with the same (df, frac, seed) -> (masked,
ground_truth, actual_fractions) shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONTINUOUS_FEATURES, PROTECTED_COLUMNS

MISSINGNESS_LEVELS = (0.10, 0.20, 0.30)


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


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.preprocessing import train_test_split_stratified

    df = load_raw_data()
    X_train, X_test, y_train, y_test = train_test_split_stratified(df)
    masked, gt, frac = mask_mcar(X_train, 0.20, seed=42)
    assert frac  # sanity: at least ran
    print("MCAR 20% actual fractions:", frac)
