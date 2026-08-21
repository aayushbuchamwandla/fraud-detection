"""
Tests for src/inference/pytorch_gpu.py.

Requires CUDA and the .venv-gpu interpreter (torch==1.12.1+cu113) -- see
that module's docstring. Skips entirely (not fails) when CUDA/GPU torch is
unavailable, so the main test suite (.venv, CPU-only torch) is unaffected.

Run with: .venv-gpu\\Scripts\\python.exe -m pytest tests\\test_gpu_inference.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "models" / "checkpoints"
HAS_CHECKPOINT = (CHECKPOINT_DIR / "fraud_mlp.pt").exists()
HAS_CUDA = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not (HAS_CUDA and HAS_CHECKPOINT),
    reason="Requires CUDA (.venv-gpu interpreter) and a trained checkpoint",
)


@pytest.fixture(scope="module")
def predictor():
    from src.inference.pytorch_gpu import GPUFraudPredictor

    return GPUFraudPredictor()


def test_predictor_loads_on_cuda(predictor):
    assert predictor.device.type == "cuda"


def test_predict_returns_correct_batch_shape(predictor):
    x = torch.randn(10, predictor.config.input_dim)
    result = predictor.predict(x)
    assert result["fraud_probability"].shape == (10,)
    assert result["is_fraud"].shape == (10,)


def test_predict_probability_in_valid_range(predictor):
    x = torch.randn(50, predictor.config.input_dim) * 5
    result = predictor.predict(x)
    probs = result["fraud_probability"]
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)


def test_gpu_and_cpu_agree_numerically():
    """The core correctness check: GPU and CPU inference on the same model
    weights and inputs must produce numerically equivalent probabilities.
    """
    from src.inference.pytorch_gpu import GPUFraudPredictor

    gpu_predictor = GPUFraudPredictor()

    # Build a CPU predictor using the same architecture/weights loaded fresh
    from src.models.model import FraudMLP

    cpu_model = FraudMLP(gpu_predictor.config)
    cpu_model.load_state_dict({k: v.cpu() for k, v in gpu_predictor.model.state_dict().items()})
    cpu_model.eval()

    torch.manual_seed(123)
    x = torch.randn(32, gpu_predictor.config.input_dim)

    gpu_result = gpu_predictor.predict(x)
    with torch.no_grad():
        cpu_probs = cpu_model.predict_proba(x)

    torch.testing.assert_close(
        gpu_result["fraud_probability"].cpu(), cpu_probs, atol=1e-5, rtol=1e-4
    )


def test_predict_single_returns_plain_python_types(predictor):
    features = [0.0] * predictor.config.input_dim
    result = predictor.predict_single(features)
    assert isinstance(result["fraud_probability"], float)
    assert isinstance(result["is_fraud"], bool)
