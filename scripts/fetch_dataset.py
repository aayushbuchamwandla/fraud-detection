"""
Download the ULB Credit Card Fraud Detection dataset via OpenML.

This is the standard, publicly-licensed dataset used throughout the project
(Pozzolo et al., ULB Machine Learning Group). It is fetched at runtime rather
than committed to the repo (see data/README.md for licensing rationale).

Usage:
    python scripts/fetch_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_PATH = RAW_DIR / "creditcard.csv"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_PATH.exists():
        print(f"Dataset already present at {OUT_PATH}, skipping download.")
        return

    print("Fetching 'creditcard' dataset (data_id=1597) from OpenML...")
    try:
        bunch = fetch_openml(data_id=1597, as_frame=True, parser="auto")
    except Exception as exc:  # noqa: BLE001
        print(f"OpenML fetch failed: {exc}", file=sys.stderr)
        print(
            "Fallback: manually download 'creditcard.csv' from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            f"and place it at {OUT_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    df = bunch.frame
    # OpenML's target column for this dataset is 'Class' with values '0'/'1' (strings)
    if df["Class"].dtype == object:
        df["Class"] = df["Class"].astype(int)

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df):,} rows x {len(df.columns)} columns to {OUT_PATH}")


if __name__ == "__main__":
    main()
