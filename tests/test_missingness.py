"""Tests for controlled MCAR masking (src/missingness.py)."""

import pandas as pd

from src.config import CONTINUOUS_FEATURES, TARGET_COLUMN
from src.data_loader import load_raw_data
from src.missingness import mask_mcar


def _sample_df():
    return load_raw_data()


def test_mcar_is_deterministic_given_seed():
    df = _sample_df()
    masked_a, gt_a, frac_a = mask_mcar(df, 0.20, seed=7)
    masked_b, gt_b, frac_b = mask_mcar(df, 0.20, seed=7)
    pd.testing.assert_frame_equal(masked_a, masked_b)
    pd.testing.assert_frame_equal(gt_a, gt_b)
    assert frac_a == frac_b


def test_target_and_protected_columns_never_masked():
    df = _sample_df()
    masked, gt, _ = mask_mcar(df, 0.30, seed=42)
    assert masked[TARGET_COLUMN].isna().sum() == 0
    assert TARGET_COLUMN not in gt.columns


def test_actual_fraction_matches_requested():
    df = _sample_df()
    _, _, actual = mask_mcar(df, 0.10, seed=42)
    for col in CONTINUOUS_FEATURES:
        assert abs(actual[col] - 0.10) < 0.01


def test_ground_truth_recovers_original_values():
    df = _sample_df()
    masked, gt, _ = mask_mcar(df, 0.20, seed=42)
    for col in CONTINUOUS_FEATURES:
        masked_idx = masked.index[masked[col].isna()]
        assert (gt.loc[masked_idx, col] == df.loc[masked_idx, col]).all()
        assert gt[col].notna().sum() == len(masked_idx)
