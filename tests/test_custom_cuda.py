"""
Correctness tests for the custom fused CUDA kernel (cuda/fraud_kernel.cu),
accessed through src/inference/custom_cuda.py.

Requires CUDA + the WSL2 ~/venv-cuda environment (torch==1.12.1+cu113 Linux
build, ninja, nvcc 11.3, g++-10) -- see README Environment notes for why
this is a separate environment from both .venv and .venv-gpu. Skips cleanly
when unavailable, same policy as test_gpu_inference.py.

Run with (inside WSL2):
    source ~/venv-cuda/bin/activate
    python -m pytest tests/test_custom_cuda.py -v
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
    reason="Requires CUDA (WSL2 ~/venv-cuda) and a trained checkpoint",
)


@pytest.fixture(scope="module")
def custom_predictor():
    from src.inference.custom_cuda import CustomCUDAPredictor

    return CustomCUDAPredictor()


@pytest.fixture(scope="module")
def cpu_model(custom_predictor):
    """A CPU FraudMLP loaded with the exact same weights as the CUDA kernel,
    used as the ground-truth reference (same role as the plain C++ reference
    in cuda/kernel_correctness_test.cu, but exercised through Python/pytest).
    """
    from src.models.model import FraudMLP, ModelConfig

    config = ModelConfig(input_dim=custom_predictor.input_dim)
    model = FraudMLP(config)
    state_dict = {
        "net.0.weight": custom_predictor.w1.cpu(),
        "net.0.bias": custom_predictor.b1.cpu(),
        "net.2.weight": custom_predictor.w2.cpu(),
        "net.2.bias": custom_predictor.b2.cpu(),
        "net.4.weight": custom_predictor.w3.cpu(),
        "net.4.bias": custom_predictor.b3.cpu(),
    }
    model.load_state_dict(state_dict)
    model.eval()
    return model


def test_predictor_loads_on_cuda(custom_predictor):
    assert custom_predictor.device.type == "cuda"


def test_output_shape(custom_predictor):
    x = torch.randn(10, custom_predictor.input_dim)
    result = custom_predictor.predict(x)
    assert result["fraud_probability"].shape == (10,)
    assert result["is_fraud"].shape == (10,)


def test_probability_in_valid_range(custom_predictor):
    x = torch.randn(200, custom_predictor.input_dim) * 5
    result = custom_predictor.predict(x)
    probs = result["fraud_probability"]
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)


@pytest.mark.parametrize("batch_size", [1, 32, 128, 512, 1024])
def test_matches_cpu_reference_numerically(custom_predictor, cpu_model, batch_size):
    """The core correctness check: the custom CUDA kernel's output must match
    a plain PyTorch CPU forward pass on identical weights/inputs, within
    float32 tolerance -- mirrors cuda/kernel_correctness_test.cu's check,
    exercised here through the actual Python inference path.
    """
    torch.manual_seed(42)
    x = torch.randn(batch_size, custom_predictor.input_dim)

    kernel_result = custom_predictor.predict(x)
    with torch.no_grad():
        cpu_probs = cpu_model.predict_proba(x)

    torch.testing.assert_close(
        kernel_result["fraud_probability"].cpu(), cpu_probs, atol=1e-5, rtol=1e-4
    )


def test_predict_single_returns_plain_python_types(custom_predictor):
    features = [0.0] * custom_predictor.input_dim
    result = custom_predictor.predict_single(features)
    assert isinstance(result["fraud_probability"], float)
    assert isinstance(result["is_fraud"], bool)
