"""
Deterministic preprocessing pipeline for the credit card fraud dataset.

Pipeline steps:
    1. Load raw CSV and validate schema.
    2. Drop exact duplicate rows (documented data quality issue in this dataset).
    3. Stratified train / val / test split (fit scaler on train only -> no leakage).
    4. Standard-scale the 'Amount' column (V1-V28 are already PCA components with
       roughly zero mean / unit variance from the original data collection, so they
       are left as-is; Amount is the one raw, unscaled feature).
    5. Save processed splits as .npy arrays for fast, dependency-light loading.

Everything here is deterministic given a fixed random seed, so re-running this
script produces byte-identical splits.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42

EXPECTED_FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"
EXPECTED_COLUMNS = set(EXPECTED_FEATURE_COLUMNS + [TARGET_COLUMN])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass
class SplitStats:
    name: str
    n_rows: int
    n_fraud: int

    @property
    def fraud_pct(self) -> float:
        return 100.0 * self.n_fraud / self.n_rows


def validate_schema(df: pd.DataFrame) -> None:
    """Raise if the dataframe doesn't match the expected schema."""
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column 'Class' contains missing values")

    unexpected_labels = set(df[TARGET_COLUMN].unique()) - {0, 1}
    if unexpected_labels:
        raise ValueError(f"Target column has unexpected labels: {unexpected_labels}")

    non_numeric = [c for c in EXPECTED_FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns: {non_numeric}")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing feature values with the train-time column median.

    The raw dataset has zero missing values (verified in scripts/inspect_dataset.py),
    but this is implemented defensively so the pipeline doesn't silently break if a
    future data refresh introduces gaps.
    """
    n_missing = int(df[EXPECTED_FEATURE_COLUMNS].isna().sum().sum())
    if n_missing == 0:
        return df
    print(f"Imputing {n_missing} missing feature values with column median")
    df = df.copy()
    for col in EXPECTED_FEATURE_COLUMNS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def load_and_clean(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found. Run scripts/fetch_dataset.py first.")

    df = pd.read_csv(raw_path)
    validate_schema(df)

    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"Dropped {n_dropped:,} exact duplicate rows ({n_before:,} -> {len(df):,})")

    df = handle_missing_values(df)
    return df


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test, preserving the fraud/legit ratio in each split."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET_COLUMN], random_state=seed
    )
    relative_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        stratify=train_val[TARGET_COLUMN],
        random_state=seed,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def scale_amount(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit a StandardScaler on Amount using TRAIN ONLY, apply to all splits."""
    scaler = StandardScaler()
    train = train.copy()
    val = val.copy()
    test = test.copy()

    train["Amount"] = scaler.fit_transform(train[["Amount"]])
    val["Amount"] = scaler.transform(val[["Amount"]])
    test["Amount"] = scaler.transform(test[["Amount"]])
    return train, val, test, scaler


def to_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = df[EXPECTED_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    return x, y


def run_pipeline(seed: int = RANDOM_SEED) -> dict:
    df = load_and_clean()
    train_df, val_df, test_df = stratified_split(df, seed=seed)
    train_df, val_df, test_df, scaler = scale_amount(train_df, val_df, test_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    splits = {}
    stats = []
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        x, y = to_arrays(split_df)
        np.save(PROCESSED_DIR / f"X_{name}.npy", x)
        np.save(PROCESSED_DIR / f"y_{name}.npy", y)
        splits[name] = (x, y)
        stats.append(SplitStats(name, len(split_df), int(split_df[TARGET_COLUMN].sum())))

    # Persist scaler parameters (not a pickle, to keep this dependency-free / auditable)
    scaler_params = {
        "feature": "Amount",
        "mean": float(scaler.mean_[0]),
        "scale": float(scaler.scale_[0]),
    }
    with open(PROCESSED_DIR / "amount_scaler.json", "w") as f:
        json.dump(scaler_params, f, indent=2)

    metadata = {
        "random_seed": seed,
        "feature_columns": EXPECTED_FEATURE_COLUMNS,
        "n_features": len(EXPECTED_FEATURE_COLUMNS),
        "splits": {s.name: {"n_rows": s.n_rows, "n_fraud": s.n_fraud, "fraud_pct": round(s.fraud_pct, 4)} for s in stats},
    }
    with open(PROCESSED_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the credit card fraud dataset")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    run_pipeline(seed=args.seed)
