"""
Correctness tests for the TensorRT engine (tensorrt/inference.py).

Loaded by explicit file path (importlib), not `from tensorrt.inference
import ...` -- this project's tensorrt/ directory shares its name with the
pip-installed `tensorrt` package, and importing it as a dotted package name
while the project root is also on sys.path risks Python resolving `import
tensorrt` (inside inference.py itself) to the wrong module. Loading by file
path avoids the ambiguity.

Requires CUDA + TensorRT + pycuda in the WSL2 ~/venv-cuda environment.
Skips cleanly when unavailable, same policy as the other GPU test files.

Run with (inside WSL2):
    source scripts/wsl_env.sh
    python -m pytest tests/test_tensorrt.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENGINE_PATH = PROJECT_ROOT / "models" / "exported" / "fraud_model.engine"

try:
    import pycuda.driver  # noqa: F401
    import tensorrt  # noqa: F401

    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False

HAS_ENGINE = ENGINE_PATH.exists()

pytestmark = pytest.mark.skipif(
    not (HAS_TENSORRT and HAS_ENGINE),
    reason="Requires TensorRT + pycuda (WSL2 ~/venv-cuda) and a built engine (tensorrt/build_engine.py)",
)


def _load_tensorrt_predictor_class():
    spec = importlib.util.spec_from_file_location("fraud_trt_inference", PROJECT_ROOT / "tensorrt" / "inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TensorRTPredictor


@pytest.fixture(scope="module")
def predictor():
    TensorRTPredictor = _load_tensorrt_predictor_class()
    return TensorRTPredictor()


@pytest.fixture(scope="module")
def cpu_model():
    """CPU reference: same architecture, weights loaded from the same checkpoint."""
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    import torch

    from src.models.model import FraudMLP, ModelConfig

    model = FraudMLP(ModelConfig(input_dim=29))
    state_dict = torch.load(PROJECT_ROOT / "models" / "checkpoints" / "fraud_mlp.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def test_engine_loads(predictor):
    assert predictor.decision_threshold > 0.0


def test_output_shape(predictor):
    import numpy as np

    x = np.random.randn(10, 29).astype(np.float32)
    result = predictor.predict(x)
    assert result["fraud_probability"].shape == (10,)
    assert result["is_fraud"].shape == (10,)


def test_probability_in_valid_range(predictor):
    import numpy as np

    x = (np.random.randn(200, 29) * 5).astype(np.float32)
    result = predictor.predict(x)
    probs = result["fraud_probability"]
    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()


@pytest.mark.parametrize("batch_size", [1, 32, 128, 512, 1024])
def test_matches_cpu_reference_numerically(predictor, cpu_model, batch_size):
    """TensorRT is *allowed* to reorder floating-point operations during its
    own kernel fusion/optimization passes, which changes rounding versus a
    straightforward PyTorch forward pass -- this is expected TensorRT
    behavior, not a bug, and the difference grows slightly with batch size
    (more reduction steps -> more opportunity for reordering). Measured on
    this model: max absolute difference ~7e-4 at batch=1024 (see
    docs/tensorrt.md), which is far too small to ever flip a classification
    decision. atol=1e-3 was chosen to clear that measured difference while
    still catching an incorrect engine (e.g. a transposed weight matrix
    would produce differences orders of magnitude larger than this).
    """
    import numpy as np
    import torch

    torch.manual_seed(42)
    x_torch = torch.randn(batch_size, 29)
    x_np = x_torch.numpy().astype(np.float32)

    trt_result = predictor.predict(x_np)
    with torch.no_grad():
        cpu_probs = cpu_model.predict_proba(x_torch).numpy()

    np.testing.assert_allclose(trt_result["fraud_probability"], cpu_probs, atol=1e-3, rtol=1e-2)


def test_predict_single_returns_plain_python_types(predictor):
    features = [0.0] * 29
    result = predictor.predict_single(features)
    assert isinstance(result["fraud_probability"], float)
    assert isinstance(result["is_fraud"], bool)


def test_real_sample_transactions_classified_correctly(predictor):
    """Uses the same labeled fixtures as the C++ demo/tests (Phase 7)."""
    import csv

    import numpy as np

    samples_path = PROJECT_ROOT / "data" / "samples" / "sample_transactions.csv"
    rows = []
    with open(samples_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)

    x = np.array([[float(v) for v in row[2:]] for row in rows], dtype=np.float32)
    true_labels = [int(row[1]) for row in rows]

    result = predictor.predict(x)
    n_correct = sum(int(is_f) == label for is_f, label in zip(result["is_fraud"], true_labels))
    assert n_correct == len(rows), f"Only {n_correct}/{len(rows)} real transactions classified correctly"
