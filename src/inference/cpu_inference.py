"""
CPU inference wrapper: loads the trained FraudMLP and exposes a clean
predict() interface used as the performance baseline for every later phase
(PyTorch GPU, custom CUDA, C++, TensorRT).
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.model import FraudMLP, ModelConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"


class CPUFraudPredictor:
    """Loads the trained model onto CPU and runs batched inference."""

    def __init__(self, checkpoint_dir: Path = CHECKPOINT_DIR):
        config_path = checkpoint_dir / "fraud_mlp_config.json"
        weights_path = checkpoint_dir / "fraud_mlp.pt"
        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"Trained model not found in {checkpoint_dir}. Run "
                "`python -m src.models.train` first."
            )

        config_dict = self._load_config_dict(config_path)
        self.decision_threshold = config_dict.pop("decision_threshold", 0.5)
        self.config = ModelConfig(**{k: v for k, v in config_dict.items() if k in ModelConfig.__dataclass_fields__})

        self.device = torch.device("cpu")
        self.model = FraudMLP(self.config).to(self.device)
        # weights_only kwarg isn't available on torch 1.12 (this project's
        # WSL2 CUDA venv -- see README Environment notes); this checkpoint
        # is just a plain state_dict of tensors regardless of which torch
        # version wrote or reads it.
        try:
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @staticmethod
    def _load_config_dict(path: Path) -> dict:
        import json

        with open(path) as f:
            return json.load(f)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        """Run inference on a batch of feature vectors.

        Args:
            x: tensor of shape (batch, n_features)

        Returns:
            dict with 'fraud_probability' (Tensor[batch]) and 'is_fraud' (Tensor[batch], bool)
        """
        x = x.to(self.device)
        probs = self.model.predict_proba(x)
        is_fraud = probs >= self.decision_threshold
        return {"fraud_probability": probs, "is_fraud": is_fraud}

    @torch.no_grad()
    def predict_single(self, features: list[float]) -> dict:
        """Convenience for a single transaction -> plain-Python result."""
        x = torch.tensor([features], dtype=torch.float32)
        result = self.predict(x)
        return {
            "fraud_probability": float(result["fraud_probability"][0]),
            "is_fraud": bool(result["is_fraud"][0]),
        }


if __name__ == "__main__":
    predictor = CPUFraudPredictor()
    print(f"Loaded model on {predictor.device}, decision threshold={predictor.decision_threshold:.4f}")

    # Smoke test on a random batch
    dummy = torch.randn(4, predictor.config.input_dim)
    result = predictor.predict(dummy)
    print("fraud_probability:", result["fraud_probability"])
    print("is_fraud:", result["is_fraud"])
