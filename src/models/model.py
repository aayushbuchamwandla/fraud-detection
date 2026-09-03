"""
PyTorch model for tabular credit-card transaction fraud classification.

Kept intentionally small (29 tabular features, two hidden layers) rather
than tuned for maximum accuracy -- a larger network would add complexity
to the CUDA kernel, C++ port, and TensorRT export without a clear benefit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    input_dim: int = 29
    hidden1: int = 64
    hidden2: int = 32

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ModelConfig":
        with open(path) as f:
            return cls(**json.load(f))


class FraudMLP(nn.Module):
    """Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> logit.

    Outputs a single raw logit (no sigmoid applied). Use `torch.sigmoid` on
    the output to get a fraud probability, or pair with `BCEWithLogitsLoss`
    during training (more numerically stable than Sigmoid + BCELoss).
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config

        self.net = nn.Sequential(
            nn.Linear(c.input_dim, c.hidden1),
            nn.ReLU(),
            nn.Linear(c.hidden1, c.hidden2),
            nn.ReLU(),
            nn.Linear(c.hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (batch, 1) -> (batch,)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience for inference: returns fraud probability in [0, 1]."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = FraudMLP()
    print(model)
    print(f"Trainable parameters: {count_parameters(model):,}")

    dummy = torch.randn(8, 29)
    logits = model(dummy)
    probs = model.predict_proba(dummy)
    print(f"logits shape: {tuple(logits.shape)}, probs shape: {tuple(probs.shape)}")
    print(f"probs range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
