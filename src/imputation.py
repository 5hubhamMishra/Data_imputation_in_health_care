"""Imputers, fit on training data only (master prompt section 21).

Two APIs live here for two different purposes:
  * `impute_mean` / `impute_median` / `impute_knn` / `impute_iterative` are
    self-contained (fit-and-transform the same df) — used to score pure
    reconstruction quality (MAE/RMSE) of a masked *training* split against
    its preserved ground truth (section 22). No test data is ever involved.
  * `fit_imputer` / `apply_imputer` are a fit/transform pair — used by the
    downstream RF-after-imputation experiment, which must fit each imputer
    on the training split only and `.transform` the held-out test split
    (section 17: no leakage from test into any fitted preprocessing step).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (registers IterativeImputer)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import CONTINUOUS_FEATURES, RANDOM_SEED

# k=5 (sklearn's default): a documented, unremarkable choice rather than a
# tuned one — with 239 training rows, 5 neighbours is a small enough
# neighbourhood to stay local without being noise-sensitive on a handful of
# points. Not CV-tuned this cycle; revisit if KNN turns out competitive.
KNN_NEIGHBORS = 5


def _make_imputer(method: str, seed: int = RANDOM_SEED):
    if method == "mean":
        return SimpleImputer(strategy="mean")
    if method == "median":
        return SimpleImputer(strategy="median")
    if method == "knn":
        return KNNImputer(n_neighbors=KNN_NEIGHBORS)
    if method == "iterative":
        return IterativeImputer(random_state=seed)
    raise ValueError(f"Unknown imputation method: {method}")


def fit_imputer(method: str, train_df: pd.DataFrame, columns: list[str] | None = None, seed: int = RANDOM_SEED):
    """Fit an imputer on `train_df` only. Pass the returned object to
    `apply_imputer` for both the training data and the held-out test data."""
    columns = columns or CONTINUOUS_FEATURES
    imputer = _make_imputer(method, seed)
    imputer.fit(train_df[columns])
    return imputer


def apply_imputer(imputer, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or CONTINUOUS_FEATURES
    out = df.copy()
    out[columns] = imputer.transform(df[columns])
    return out


def impute_mean(masked_df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    imputer = fit_imputer("mean", masked_df, columns)
    return apply_imputer(imputer, masked_df, columns)


def impute_median(masked_df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    imputer = fit_imputer("median", masked_df, columns)
    return apply_imputer(imputer, masked_df, columns)


def impute_knn(masked_df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    imputer = fit_imputer("knn", masked_df, columns)
    return apply_imputer(imputer, masked_df, columns)


def impute_iterative(masked_df: pd.DataFrame, columns: list[str] | None = None, seed: int = RANDOM_SEED) -> pd.DataFrame:
    imputer = fit_imputer("iterative", masked_df, columns, seed=seed)
    return apply_imputer(imputer, masked_df, columns)


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
