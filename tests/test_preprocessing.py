"""
Tests for src/data/preprocessing.py.

Run with: pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    EXPECTED_FEATURE_COLUMNS,
    TARGET_COLUMN,
    handle_missing_values,
    scale_amount,
    stratified_split,
    to_arrays,
    validate_schema,
)


def make_toy_df(n_rows: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(size=n_rows) for col in EXPECTED_FEATURE_COLUMNS}
    # ~1% fraud rate, deterministic
    labels = np.zeros(n_rows, dtype=int)
    fraud_idx = rng.choice(n_rows, size=max(2, n_rows // 100), replace=False)
    labels[fraud_idx] = 1
    data[TARGET_COLUMN] = labels
    return pd.DataFrame(data)


def test_validate_schema_accepts_valid_df():
    df = make_toy_df()
    validate_schema(df)  # should not raise


def test_validate_schema_rejects_missing_column():
    df = make_toy_df().drop(columns=["V1"])
    with pytest.raises(ValueError, match="Missing expected columns"):
        validate_schema(df)


def test_validate_schema_rejects_bad_labels():
    df = make_toy_df()
    df.loc[0, TARGET_COLUMN] = 2
    with pytest.raises(ValueError, match="unexpected labels"):
        validate_schema(df)


def test_validate_schema_rejects_non_numeric_feature():
    df = make_toy_df()
    df["V1"] = df["V1"].astype(str)
    with pytest.raises(ValueError, match="Non-numeric"):
        validate_schema(df)


def test_handle_missing_values_imputes_median():
    df = make_toy_df()
    df.loc[0, "V1"] = np.nan
    median_v1 = df["V1"].median()  # median of the column WITH the NaN (pandas skips it)
    out = handle_missing_values(df)
    assert out.loc[0, "V1"] == pytest.approx(median_v1)
    assert out["V1"].isna().sum() == 0


def test_handle_missing_values_noop_when_no_nans():
    df = make_toy_df()
    out = handle_missing_values(df)
    pd.testing.assert_frame_equal(df, out)


def test_stratified_split_preserves_fraud_ratio():
    df = make_toy_df(n_rows=2000, seed=1)
    overall_ratio = df[TARGET_COLUMN].mean()

    train, val, test = stratified_split(df, test_size=0.15, val_size=0.15, seed=42)

    assert len(train) + len(val) + len(test) == len(df)
    for split in (train, val, test):
        ratio = split[TARGET_COLUMN].mean()
        assert ratio == pytest.approx(overall_ratio, abs=0.02)


def test_stratified_split_is_deterministic():
    df = make_toy_df(n_rows=1000, seed=2)
    train1, val1, test1 = stratified_split(df, seed=42)
    train2, val2, test2 = stratified_split(df, seed=42)

    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(val1, val2)
    pd.testing.assert_frame_equal(test1, test2)


def test_stratified_split_no_row_overlap():
    df = make_toy_df(n_rows=1000, seed=3).reset_index(drop=True)
    df["_row_id"] = df.index
    train, val, test = stratified_split(df, seed=42)

    train_ids = set(train["_row_id"])
    val_ids = set(val["_row_id"])
    test_ids = set(test["_row_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_scale_amount_fits_on_train_only_no_leakage():
    df = make_toy_df(n_rows=1000, seed=4)
    df["Amount"] = np.random.default_rng(5).uniform(0, 1000, size=len(df))
    train, val, test = stratified_split(df, seed=42)

    train_scaled, val_scaled, test_scaled, scaler = scale_amount(train, val, test)

    # Scaler statistics must come from train only
    assert scaler.mean_[0] == pytest.approx(train["Amount"].mean())
    assert scaler.scale_[0] == pytest.approx(train["Amount"].std(ddof=0))

    # Scaled train Amount should have ~zero mean, unit variance
    assert train_scaled["Amount"].mean() == pytest.approx(0.0, abs=1e-8)
    assert train_scaled["Amount"].std(ddof=0) == pytest.approx(1.0, abs=1e-8)

    # Val/test are transformed with train statistics, not refit -> generally
    # will NOT have exactly zero mean/unit variance
    val_manual = (val["Amount"] - scaler.mean_[0]) / scaler.scale_[0]
    np.testing.assert_allclose(val_scaled["Amount"].to_numpy(), val_manual.to_numpy(), rtol=1e-6)


def test_to_arrays_shapes_and_dtypes():
    df = make_toy_df(n_rows=50, seed=6)
    x, y = to_arrays(df)
    assert x.shape == (50, len(EXPECTED_FEATURE_COLUMNS))
    assert y.shape == (50,)
    assert x.dtype == np.float32
    assert y.dtype == np.float32
