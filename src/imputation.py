"""Classical imputers, fit on training data only (master prompt section 21).

Only mean/median are implemented this cycle. KNN and an ML-based imputer
(IterativeImputer or similar) are added as sibling `impute_<method>`
functions with the same (masked_df) -> imputed_df shape later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import CONTINUOUS_FEATURES


def impute_mean(masked_df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or CONTINUOUS_FEATURES
    imputed = masked_df.copy()
    for col in columns:
        imputed[col] = imputed[col].fillna(imputed[col].mean())
    return imputed


def impute_median(masked_df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or CONTINUOUS_FEATURES
    imputed = masked_df.copy()
    for col in columns:
        imputed[col] = imputed[col].fillna(imputed[col].median())
    return imputed


def score_imputation(imputed_df: pd.DataFrame, ground_truth: pd.DataFrame) -> dict[str, dict[str, float]]:
    """MAE/RMSE per column, computed only on the intentionally hidden cells
    recorded in `ground_truth` (section 22 — never score naturally-observed
    cells, only the masked ones)."""
    scores: dict[str, dict[str, float]] = {}
    for col in ground_truth.columns:
        hidden_idx = ground_truth.index[ground_truth[col].notna()]
        if len(hidden_idx) == 0:
            continue
        y_true = ground_truth.loc[hidden_idx, col]
        y_pred = imputed_df.loc[hidden_idx, col]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        scores[col] = {"mae": mae, "rmse": rmse}
    return scores
