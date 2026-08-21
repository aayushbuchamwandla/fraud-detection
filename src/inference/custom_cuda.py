"""
Custom fused-CUDA-kernel inference wrapper -- same predict() interface as
cpu_inference.py / pytorch_gpu.py, backed by the hand-written kernel in
cuda/fraud_kernel.cu instead of PyTorch's eager-mode op dispatch.

Must run inside WSL2 with a CUDA-enabled torch build (this project's
~/venv-cuda) -- JIT-compiling a CUDA extension requires nvcc + a compatible
host compiler, which native Windows could not provide for this driver's
CUDA 11.3 ceiling (see docs/bottleneck_analysis.md and the README's
Environment notes for why WSL2 was needed starting at this phase).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
CUDA_DIR = PROJECT_ROOT / "cuda"

# nvcc 11.3 requires an older host compiler than this system's default gcc-11
# (see README Environment notes) -- g++-10 is installed alongside specifically
# for this.
os.environ.setdefault("CC", "/usr/bin/gcc-10")
os.environ.setdefault("CXX", "/usr/bin/g++-10")

_extension = None


def _get_extension():
    """JIT-compile the CUDA extension on first use (cached by torch after that)."""
    global _extension
    if _extension is None:
        _extension = load(
            name="fraud_cuda_kernel",
            sources=[str(CUDA_DIR / "torch_binding.cpp"), str(CUDA_DIR / "fraud_kernel.cu")],
            extra_cuda_cflags=["-arch=sm_86", "-ccbin=/usr/bin/g++-10", "-O3"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _extension


class CustomCUDAPredictor:
    """Loads trained weights onto CUDA and runs inference via the custom fused kernel."""

    def __init__(self, checkpoint_dir: Path = CHECKPOINT_DIR):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Run inside WSL2 with ~/venv-cuda active.")

        config_path = checkpoint_dir / "fraud_mlp_config.json"
        weights_path = checkpoint_dir / "fraud_mlp.pt"
        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"Trained model not found in {checkpoint_dir}. Run `python -m src.models.train` first."
            )

        with open(config_path) as f:
            config_dict = json.load(f)
        self.decision_threshold = config_dict.get("decision_threshold", 0.5)
        self.input_dim = config_dict["input_dim"]

        self.device = torch.device("cuda")
        state_dict = torch.load(weights_path, map_location=self.device)

        # nn.Sequential indices: 0=Linear(29,64), 2=Linear(64,32), 4=Linear(32,1)
        self.w1 = state_dict["net.0.weight"].to(self.device).contiguous()
        self.b1 = state_dict["net.0.bias"].to(self.device).contiguous()
        self.w2 = state_dict["net.2.weight"].to(self.device).contiguous()
        self.b2 = state_dict["net.2.bias"].to(self.device).contiguous()
        self.w3 = state_dict["net.4.weight"].to(self.device).contiguous()
        self.b3 = state_dict["net.4.bias"].to(self.device).contiguous()

        self.ext = _get_extension()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        x = x.to(self.device).float()
        probs = self.ext.fraud_mlp_forward(x, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3)
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

    predictor = CustomCUDAPredictor()
    print(f"Model loaded on {predictor.device}, decision threshold={predictor.decision_threshold:.4f}")

    dummy = torch.randn(4, predictor.input_dim)
    result = predictor.predict(dummy)
    print("fraud_probability:", result["fraud_probability"].cpu())
    print("is_fraud:", result["is_fraud"].cpu())
