"""
Export the trained FraudMLP's weights as raw float32 binary files.

Used by the standalone C++/CUDA build (cuda/, cpp/) so those components don't
need to link against libtorch just to read a checkpoint -- they read flat
binary arrays instead.

Usage:
    python scripts/export_weights.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
EXPORT_DIR = PROJECT_ROOT / "models" / "exported"

# nn.Sequential indices: 0=Linear(29,64), 2=Linear(64,32), 4=Linear(32,1)
TENSOR_NAMES = {
    "net.0.weight": "w1.bin",
    "net.0.bias": "b1.bin",
    "net.2.weight": "w2.bin",
    "net.2.bias": "b2.bin",
    "net.4.weight": "w3.bin",
    "net.4.bias": "b3.bin",
}


def main() -> None:
    weights_path = CHECKPOINT_DIR / "fraud_mlp.pt"
    config_path = CHECKPOINT_DIR / "fraud_mlp_config.json"
    if not weights_path.exists():
        raise FileNotFoundError(f"{weights_path} not found. Run `python -m src.models.train` first.")

    state_dict = torch.load(weights_path, map_location="cpu")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    shapes = {}
    for tensor_name, filename in TENSOR_NAMES.items():
        arr = state_dict[tensor_name].numpy().astype(np.float32)
        arr.tofile(EXPORT_DIR / filename)
        shapes[filename] = list(arr.shape)
        print(f"{tensor_name:16s} -> {filename:10s} shape={arr.shape}")

    import json

    with open(EXPORT_DIR / "shapes.json", "w") as f:
        json.dump(shapes, f, indent=2)

    # Plain-text threshold file (not JSON) so the C++ component doesn't need
    # a JSON library just to read one float.
    if config_path.exists():
        with open(config_path) as f:
            threshold = json.load(f).get("decision_threshold", 0.5)
    else:
        threshold = 0.5
    with open(EXPORT_DIR / "threshold.txt", "w") as f:
        f.write(f"{threshold}\n")
    print(f"decision_threshold -> threshold.txt = {threshold}")

    print(f"\nExported to {EXPORT_DIR}")


if __name__ == "__main__":
    main()
