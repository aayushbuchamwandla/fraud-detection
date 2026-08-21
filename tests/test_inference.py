"""
Tests for src/inference/cpu_inference.py.

Requires a trained model checkpoint (run `python -m src.models.train` first).
Skips gracefully if no checkpoint is present, so this doesn't break CI on a
fresh clone before training has been run.

Run with: pytest tests/test_inference.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "models" / "checkpoints"
HAS_CHECKPOINT = (CHECKPOINT_DIR / "fraud_mlp.pt").exists()

pytestmark = pytest.mark.skipif(
    not HAS_CHECKPOINT, reason="No trained model checkpoint; run `python -m src.models.train` first"
)


@pytest.fixture(scope="module")
def predictor():
    from src.inference.cpu_inference import CPUFraudPredictor

    return CPUFraudPredictor()


def test_predictor_loads_on_cpu(predictor):
    assert predictor.device.type == "cpu"


def test_predictor_decision_threshold_in_range(predictor):
    assert 0.0 <= predictor.decision_threshold <= 1.0


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


def test_is_fraud_matches_threshold(predictor):
    x = torch.randn(50, predictor.config.input_dim) * 5
    result = predictor.predict(x)
    expected = result["fraud_probability"] >= predictor.decision_threshold
    torch.testing.assert_close(result["is_fraud"], expected)


def test_predict_single_returns_plain_python_types(predictor):
    features = [0.0] * predictor.config.input_dim
    result = predictor.predict_single(features)
    assert isinstance(result["fraud_probability"], float)
    assert isinstance(result["is_fraud"], bool)


def test_predict_is_deterministic(predictor):
    x = torch.randn(5, predictor.config.input_dim)
    r1 = predictor.predict(x)
    r2 = predictor.predict(x)
    torch.testing.assert_close(r1["fraud_probability"], r2["fraud_probability"])
