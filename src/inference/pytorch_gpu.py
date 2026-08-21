"""
PyTorch CUDA inference wrapper -- the GPU counterpart to cpu_inference.py.

Environment note: this project's driver (471.41, CUDA 11.4 ceiling) predates
the CUDA 12.x builds that current PyTorch/Python 3.13 require. Rather than
require a driver update, this module runs under a separate Python 3.10 venv
(.venv-gpu) with torch==1.12.1+cu113 -- a build whose CUDA 11.3 runtime is
covered by the installed driver's CUDA 11.4 ceiling via NVIDIA's minor-version
compatibility guarantee within the CUDA 11 major series. Model checkpoints
are saved as plain state_dict tensors (see src/models/train.py), which load
identically regardless of which torch version wrote or reads them, so the
same fraud_mlp.pt trained under torch 2.13/CPU loads correctly here.

Run this module's benchmark with:
    .venv-gpu\\Scripts\\python.exe -m src.inference.pytorch_gpu
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.models.model import FraudMLP, ModelConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"


class GPUFraudPredictor:
    """Loads the trained model onto CUDA and runs batched inference."""

    def __init__(self, checkpoint_dir: Path = CHECKPOINT_DIR):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available in this environment. This module requires "
                "the .venv-gpu interpreter (torch==1.12.1+cu113) -- see module docstring."
            )

        config_path = checkpoint_dir / "fraud_mlp_config.json"
        weights_path = checkpoint_dir / "fraud_mlp.pt"
        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"Trained model not found in {checkpoint_dir}. Run "
                "`python -m src.models.train` first (CPU venv is fine for training)."
            )

        with open(config_path) as f:
            config_dict = json.load(f)
        self.decision_threshold = config_dict.pop("decision_threshold", 0.5)
        self.config = ModelConfig(**{k: v for k, v in config_dict.items() if k in ModelConfig.__dataclass_fields__})

        self.device = torch.device("cuda")
        self.model = FraudMLP(self.config).to(self.device)
        # weights_only kwarg not available on torch 1.12; state_dict tensors
        # are the only thing in this checkpoint file regardless.
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        x = x.to(self.device)
        probs = self.model.predict_proba(x)
        is_fraud = probs >= self.decision_threshold
        return {"fraud_probability": probs, "is_fraud": is_fraud}

    @torch.no_grad()
    def predict_single(self, features: list[float]) -> dict:
        x = torch.tensor([features], dtype=torch.float32)
        result = self.predict(x)
        return {
            "fraud_probability": float(result["fraud_probability"][0].cpu()),
            "is_fraud": bool(result["is_fraud"][0].cpu()),
        }


if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"torch version: {torch.__version__} (built for CUDA {torch.version.cuda})")

    predictor = GPUFraudPredictor()
    print(f"Model loaded on {predictor.device}, decision threshold={predictor.decision_threshold:.4f}")

    dummy = torch.randn(4, predictor.config.input_dim)
    result = predictor.predict(dummy)
    print("fraud_probability:", result["fraud_probability"].cpu())
    print("is_fraud:", result["is_fraud"].cpu())
