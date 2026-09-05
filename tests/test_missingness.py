"""Tests for controlled MCAR/MAR/MNAR masking (src/missingness.py)."""

import pandas as pd
from scipy.stats import pointbiserialr

from src.config import CONTINUOUS_FEATURES, TARGET_COLUMN
from src.data_loader import load_raw_data
from src.missingness import (
    MAR_CONDITIONING, MNAR_CONDITIONING, mask_mar, mask_mcar, mask_mnar,
)


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


def test_mar_target_and_protected_columns_never_masked():
    df = _sample_df()
    masked, gt, _ = mask_mar(df, 0.30, seed=42)
    assert masked[TARGET_COLUMN].isna().sum() == 0
    assert TARGET_COLUMN not in gt.columns


def test_mar_actual_fraction_matches_requested():
    df = _sample_df()
    _, _, actual = mask_mar(df, 0.20, seed=42)
    for col in MAR_CONDITIONING:
        assert abs(actual[col] - 0.20) < 0.01


def test_mar_ground_truth_recovers_original_values():
    df = _sample_df()
    masked, gt, _ = mask_mar(df, 0.20, seed=42)
    for col in MAR_CONDITIONING:
        masked_idx = masked.index[masked[col].isna()]
        assert (gt.loc[masked_idx, col] == df.loc[masked_idx, col]).all()


def test_mar_missingness_correlates_with_conditioning_column():
    """Sanity check that MAR is actually conditioned as documented, not
    uniform random masking mislabeled as MAR: the missingness indicator for
    each key column must correlate with its conditioning column in the
    documented direction."""
    df = _sample_df()
    masked, _, _ = mask_mar(df, 0.30, seed=42)
    for col, (cond_col, favor) in MAR_CONDITIONING.items():
        is_missing = masked[col].isna().astype(int)
        corr, pvalue = pointbiserialr(is_missing, df[cond_col])
        expected_sign = 1 if favor == "high" else -1
        assert corr * expected_sign > 0
        assert pvalue < 0.05


def test_mnar_target_and_protected_columns_never_masked():
    df = _sample_df()
    masked, gt, _ = mask_mnar(df, 0.30, seed=42)
    assert masked[TARGET_COLUMN].isna().sum() == 0
    assert TARGET_COLUMN not in gt.columns


def test_mnar_actual_fraction_matches_requested():
    df = _sample_df()
    _, _, actual = mask_mnar(df, 0.20, seed=42)
    for col in MNAR_CONDITIONING:
        assert abs(actual[col] - 0.20) < 0.01


def test_mnar_ground_truth_recovers_original_values():
    df = _sample_df()
    masked, gt, _ = mask_mnar(df, 0.20, seed=42)
    for col in MNAR_CONDITIONING:
        masked_idx = masked.index[masked[col].isna()]
        assert (gt.loc[masked_idx, col] == df.loc[masked_idx, col]).all()


def test_mnar_missingness_correlates_with_own_value():
    """Sanity check that MNAR missingness actually depends on the masked
    value itself, in the documented direction."""
    df = _sample_df()
    masked, _, _ = mask_mnar(df, 0.30, seed=42)
    for col, favor in MNAR_CONDITIONING.items():
        is_missing = masked[col].isna().astype(int)
        corr, pvalue = pointbiserialr(is_missing, df[col])
        expected_sign = 1 if favor == "high" else -1
        assert corr * expected_sign > 0
        assert pvalue < 0.05
