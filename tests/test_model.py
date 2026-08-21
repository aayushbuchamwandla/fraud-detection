"""
Tests for src/models/model.py and evaluation helpers in src/models/train.py.

Run with: pytest tests/test_model.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.model import FraudMLP, ModelConfig, count_parameters
from src.models.train import best_f1_threshold, metrics_at_threshold


def test_model_output_shape():
    model = FraudMLP(ModelConfig(input_dim=29))
    x = torch.randn(16, 29)
    logits = model(x)
    assert logits.shape == (16,)


def test_model_single_sample_shape():
    model = FraudMLP(ModelConfig(input_dim=29))
    x = torch.randn(1, 29)
    logits = model(x)
    assert logits.shape == (1,)


def test_predict_proba_in_valid_range():
    model = FraudMLP(ModelConfig(input_dim=29))
    x = torch.randn(100, 29) * 10  # wide range of inputs, including extremes
    probs = model.predict_proba(x)
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)


def test_predict_proba_matches_sigmoid_of_logits():
    model = FraudMLP(ModelConfig(input_dim=29))
    x = torch.randn(8, 29)
    with torch.no_grad():
        logits = model(x)
    probs = model.predict_proba(x)
    torch.testing.assert_close(probs, torch.sigmoid(logits))


def test_model_deterministic_given_same_weights():
    model = FraudMLP(ModelConfig(input_dim=29))
    model.eval()
    x = torch.randn(4, 29)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    torch.testing.assert_close(out1, out2)


def test_model_respects_custom_config():
    config = ModelConfig(input_dim=10, hidden1=8, hidden2=4)
    model = FraudMLP(config)
    x = torch.randn(3, 10)
    logits = model(x)
    assert logits.shape == (3,)
    # First layer must match input_dim
    first_linear = model.net[0]
    assert first_linear.in_features == 10
    assert first_linear.out_features == 8


def test_count_parameters_positive():
    model = FraudMLP(ModelConfig(input_dim=29))
    assert count_parameters(model) > 0


def test_model_config_roundtrip(tmp_path):
    config = ModelConfig(input_dim=29, hidden1=64, hidden2=32)
    path = tmp_path / "config.json"
    config.save(path)
    loaded = ModelConfig.load(path)
    assert loaded == config


def test_checkpoint_save_load_roundtrip(tmp_path):
    model = FraudMLP(ModelConfig(input_dim=29))
    model.eval()
    x = torch.randn(5, 29)
    with torch.no_grad():
        original_output = model(x)

    ckpt_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt_path)

    loaded_model = FraudMLP(ModelConfig(input_dim=29))
    # weights_only kwarg isn't available on torch 1.12 (this project's WSL2
    # CUDA venv -- see README Environment notes); fall back gracefully.
    try:
        state_dict = torch.load(ckpt_path, weights_only=True)
    except TypeError:
        state_dict = torch.load(ckpt_path)
    loaded_model.load_state_dict(state_dict)
    loaded_model.eval()
    with torch.no_grad():
        loaded_output = loaded_model(x)

    torch.testing.assert_close(original_output, loaded_output)


# --- threshold selection / metrics helpers (src/models/train.py) ---


def test_best_f1_threshold_perfect_separation():
    # Labels perfectly separated by probability: fraud always > 0.9, legit always < 0.1
    labels = np.array([0, 0, 0, 1, 1])
    probs = np.array([0.05, 0.1, 0.2, 0.95, 0.99])
    threshold = best_f1_threshold(probs, labels)
    # Any threshold in (0.2, 0.95] achieves perfect F1=1.0
    assert 0.2 < threshold <= 0.95


def test_metrics_at_threshold_confusion_matrix_sums_to_n():
    labels = np.array([0, 0, 1, 1, 0, 1])
    probs = np.array([0.1, 0.4, 0.9, 0.6, 0.3, 0.2])
    m = metrics_at_threshold(probs, labels, threshold=0.5)
    cm = m["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(labels)
    assert m["n_samples"] == len(labels)
    assert m["n_fraud"] == int(labels.sum())


def test_metrics_at_threshold_all_correct():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.01, 0.02, 0.99, 0.98])
    m = metrics_at_threshold(probs, labels, threshold=0.5)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
