"""Train/test split. Must run before any imputer/scaler/selector is fit
(master prompt section 17: split before fitting anything, to prevent
leakage from the held-out test set)."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, TARGET_COLUMN, TEST_SIZE


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def train_test_split_stratified(df: pd.DataFrame, seed: int = RANDOM_SEED):
    """80/20 stratified split on the target. `seed` defaults to the project
    seed but is overridable for repeated-seed experiments (master prompt
    section 29)."""
    X, y = split_features_target(df)
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
