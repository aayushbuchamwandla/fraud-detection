"""
Phase 8, step 1: export the trained FraudMLP to ONNX.

ONNX is the standard intermediate format between PyTorch and TensorRT --
using it means the TensorRT build step doesn't need to know anything about
PyTorch's internal module structure, and the same .onnx file would work
with any other ONNX-consuming runtime too.

Usage:
    python tensorrt/export_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model import FraudMLP, ModelConfig  # noqa: E402

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
EXPORT_DIR = PROJECT_ROOT / "models" / "exported"


class FraudMLPWithSigmoid(torch.nn.Module):
    """Wraps FraudMLP to output a probability, not a raw logit.

    FraudMLP.forward() intentionally returns logits (for BCEWithLogitsLoss
    during training -- see src/models/model.py). Every other inference
    backend (CPU, PyTorch GPU, custom CUDA kernel, C++) applies sigmoid as
    an explicit separate step in its predict() wrapper.
    For ONNX/TensorRT, baking sigmoid into the exported graph keeps the
    engine's output directly comparable/pluggable the same way, and lets
    TensorRT fuse it into the same kernel as the final Linear layer.
    """

    def __init__(self, base_model: FraudMLP):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.base_model(x))


def main() -> None:
    weights_path = CHECKPOINT_DIR / "fraud_mlp.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"{weights_path} not found. Run `python -m src.models.train` first.")

    base_model = FraudMLP(ModelConfig(input_dim=29))
    base_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    base_model.eval()
    model = FraudMLPWithSigmoid(base_model)
    model.eval()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = EXPORT_DIR / "fraud_model.onnx"

    # Dynamic batch dimension so the TensorRT engine can be built for a
    # range of batch sizes (see build_engine.py's optimization profile),
    # matching the same batch sizes used everywhere else in this project.
    dummy_input = torch.randn(1, 29)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["features"],
        output_names=["fraud_probability"],
        dynamic_axes={"features": {0: "batch_size"}, "fraud_probability": {0: "batch_size"}},
        opset_version=13,  # torch==1.12.1 (see README Environment notes) supports up to opset 16;
        # 13 is used for broad compatibility with the TensorRT ONNX parser version installed.
    )

    print(f"Exported ONNX model to {onnx_path} (sigmoid included -- outputs probabilities, not logits)")

    try:
        import onnx

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model structure is valid (onnx.checker passed).")
        print(f"Graph inputs: {[i.name for i in onnx_model.graph.input]}")
        print(f"Graph outputs: {[o.name for o in onnx_model.graph.output]}")
    except ImportError:
        print("(onnx package not installed -- skipping structural validation)")


if __name__ == "__main__":
    main()
