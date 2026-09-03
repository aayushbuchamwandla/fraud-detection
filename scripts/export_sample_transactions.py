"""
Export a small, fixed set of transactions (with true labels) from the
held-out test split, used as fixtures by the C++ demo, terminal demo, API
tests, and web demo.

Usage:
    python scripts/export_sample_transactions.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"

N_LEGIT = 5
N_FRAUD = 5
SEED = 42


def main() -> None:
    x_path = PROCESSED_DIR / "X_test.npy"
    y_path = PROCESSED_DIR / "y_test.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"{x_path} not found. Run `python -m src.data.preprocessing` first.")

    x = np.load(x_path)
    y = np.load(y_path)

    rng = np.random.default_rng(SEED)
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]

    chosen_fraud = rng.choice(fraud_idx, size=min(N_FRAUD, len(fraud_idx)), replace=False)
    chosen_legit = rng.choice(legit_idx, size=min(N_LEGIT, len(legit_idx)), replace=False)
    chosen = np.concatenate([chosen_legit, chosen_fraud])

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAMPLES_DIR / "sample_transactions.csv"

    n_features = x.shape[1]
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    assert len(feature_cols) == n_features

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_split_index", "true_label", *feature_cols])
        for idx in chosen:
            writer.writerow([int(idx), int(y[idx]), *[f"{v:.8f}" for v in x[idx]]])

    print(f"Exported {len(chosen)} transactions ({len(chosen_legit)} legitimate, {len(chosen_fraud)} fraud)")
    print(f"-> {out_path}")
    print("\nSource: held-out test split (data/processed/X_test.npy, y_test.npy), seed=42.")
    print("These are real transactions with real ground-truth labels from the ULB dataset --")
    print("not synthetic or invented examples.")


if __name__ == "__main__":
    main()
