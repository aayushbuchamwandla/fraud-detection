"""
Tests for src/api/server.py.

Uses FastAPI's TestClient (no real network/socket needed) and the real
trained model + real labeled sample transactions (data/samples/sample_transactions.csv),
not fabricated fixtures.

Run with: pytest tests/test_api.py -v
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
SAMPLES_PATH = PROJECT_ROOT / "data" / "samples" / "sample_transactions.csv"

HAS_CHECKPOINT = (CHECKPOINT_DIR / "fraud_mlp.pt").exists()
HAS_SAMPLES = SAMPLES_PATH.exists()

pytestmark = pytest.mark.skipif(
    not HAS_CHECKPOINT, reason="Requires a trained checkpoint; run `python -m src.models.train` first"
)


@pytest.fixture(scope="module")
def client():
    from src.api.server import create_app

    app = create_app()
    return TestClient(app)


def load_sample_transactions() -> list[dict]:
    if not HAS_SAMPLES:
        return []
    with open(SAMPLES_PATH) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for row in reader:
            rows.append({"true_label": int(row[1]), "features": [float(v) for v in row[2:]]})
        return rows


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["engine"] == "cpu"
    assert 0.0 < body["decision_threshold"] < 1.0


def test_predict_valid_request(client):
    features = [0.0] * 29
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    body = response.json()
    assert "fraud_probability" in body
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)
    assert body["engine"] == "cpu"
    assert body["latency_ms"] > 0.0


def test_predict_wrong_feature_count_returns_422(client):
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})  # only 3, need 29
    assert response.status_code == 422


def test_predict_missing_features_field_returns_422(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_non_numeric_features_returns_422(client):
    response = client.post("/predict", json={"features": ["not", "a", "number"] + [0.0] * 26})
    assert response.status_code == 422


def test_predict_malformed_json_returns_422(client):
    response = client.post("/predict", content="{not valid json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422


@pytest.mark.skipif(not HAS_SAMPLES, reason="Requires data/samples/sample_transactions.csv")
def test_predict_real_sample_transactions_all_correct(client):
    """Same real, labeled fixtures used by the C++ (Phase 7) and TensorRT
    (Phase 8) correctness tests -- not synthetic data."""
    samples = load_sample_transactions()
    assert len(samples) > 0

    n_correct = 0
    for sample in samples:
        response = client.post("/predict", json={"features": sample["features"]})
        assert response.status_code == 200
        body = response.json()
        n_correct += int(body["is_fraud"]) == sample["true_label"]

    assert n_correct == len(samples), f"Only {n_correct}/{len(samples)} real transactions classified correctly"
