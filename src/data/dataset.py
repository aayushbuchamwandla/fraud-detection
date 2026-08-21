"""
PyTorch Dataset / DataLoader helpers for the processed fraud dataset.

Reads the .npy arrays produced by src/data/preprocessing.py and exposes them
as a torch Dataset, with a small helper for building DataLoaders with a
consistent batch size across training and benchmarking code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class FraudDataset(Dataset):
    """Wraps the preprocessed (X, y) numpy arrays for a single split."""

    def __init__(self, split: str, processed_dir: Path = PROCESSED_DIR):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got {split!r}")

        x_path = processed_dir / f"X_{split}.npy"
        y_path = processed_dir / f"y_{split}.npy"
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Processed data not found for split={split!r} in {processed_dir}. "
                "Run `python -m src.data.preprocessing` first."
            )

        x = np.load(x_path)
        y = np.load(y_path)
        assert len(x) == len(y), "Feature/label length mismatch"

        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        self.n_features = self.x.shape[1]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def get_dataloader(
    split: str,
    batch_size: int = 256,
    shuffle: bool | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for the given split.

    Shuffling defaults to True for train, False for val/test (deterministic
    evaluation order matters for benchmarking / reproducible metrics).
    """
    dataset = FraudDataset(split)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        ds = FraudDataset(split)
        loader = get_dataloader(split, batch_size=256)
        xb, yb = next(iter(loader))
        print(f"{split}: {len(ds):,} rows, {ds.n_features} features, batch shape {tuple(xb.shape)}, label shape {tuple(yb.shape)}")
