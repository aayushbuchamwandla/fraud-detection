"""
Inspect the raw credit card fraud dataset and print a factual summary.

Usage:
    python scripts/inspect_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "creditcard.csv"


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_PATH} not found. Run scripts/fetch_dataset.py first."
        )

    df = pd.read_csv(RAW_PATH)

    n_total = len(df)
    n_fraud = int((df["Class"] == 1).sum())
    n_legit = n_total - n_fraud
    fraud_pct = 100 * n_fraud / n_total

    n_features = df.shape[1] - 1  # exclude target column 'Class'
    n_missing = int(df.isna().sum().sum())
    n_duplicates = int(df.duplicated().sum())
    has_time = "Time" in df.columns

    print("=" * 60)
    print("DATASET SUMMARY: creditcard.csv (ULB, via OpenML data_id=1597)")
    print("=" * 60)
    print(f"Total transactions:        {n_total:,}")
    print(f"Legitimate transactions:   {n_legit:,}")
    print(f"Fraudulent transactions:   {n_fraud:,}")
    print(f"Fraud percentage:          {fraud_pct:.4f}%")
    print(f"Class imbalance ratio:     1 : {n_legit / n_fraud:.1f} (fraud:legit)")
    feature_desc = "Time, V1-V28, Amount" if has_time else "V1-V28, Amount (no Time column in this source)"
    print(f"Number of features:        {n_features} ({feature_desc})")
    print(f"Missing values (total):    {n_missing}")
    print(f"Duplicated rows:           {n_duplicates:,}")
    print()
    print("Columns:", list(df.columns))
    print()
    print("dtypes:")
    print(df.dtypes)
    print()
    print("Amount statistics:")
    print(df["Amount"].describe())
    print()
    if has_time:
        print("Time statistics (seconds elapsed from first transaction):")
        print(df["Time"].describe())
    else:
        print(
            "Note: this OpenML mirror (data_id=1597) omits the original 'Time' "
            "column present in the Kaggle release. Time is not used as a model "
            "feature in this project, so this has no effect on preprocessing."
        )


if __name__ == "__main__":
    main()
