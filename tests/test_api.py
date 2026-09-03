"""
Tests for src/api/server.py.

Uses FastAPI's TestClient (no network socket needed), the trained model,
and the labeled sample transactions in data/samples/sample_transactions.csv.

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


def test_feature_names_endpoint(client):
    response = client.get("/feature-names")
    assert response.status_code == 200
    names = response.json()["feature_names"]
    assert len(names) == 29
    assert names[0] == "V1"
    assert names[-1] == "Amount"


@pytest.mark.skipif(not HAS_SAMPLES, reason="Requires data/samples/sample_transactions.csv")
def test_samples_endpoint_returns_real_labeled_data(client):
    response = client.get("/samples")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 10
    assert all(len(row["features"]) == 29 for row in data)
    assert any(row["true_label"] == 1 for row in data)
    assert any(row["true_label"] == 0 for row in data)


def test_frontend_served_at_root(client):
    """The frontend page is served by the same API it calls."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "GPU-Accelerated Fraud Detection" in response.text


@pytest.mark.skipif(not HAS_SAMPLES, reason="Requires data/samples/sample_transactions.csv")
def test_full_frontend_flow_sample_to_prediction(client):
    """Exercises the sequence the web page's JS performs when a user clicks
    'Load Fraud Example' then 'Run Inference': GET /samples, pick a fraud
    row, POST its features to /predict."""
    samples = client.get("/samples").json()
    fraud_sample = next(s for s in samples if s["true_label"] == 1)

    response = client.post("/predict", json={"features": fraud_sample["features"]})
    assert response.status_code == 200
    body = response.json()
    assert body["is_fraud"] is True
    assert body["fraud_probability"] > 0.5


@pytest.mark.skipif(not HAS_SAMPLES, reason="Requires data/samples/sample_transactions.csv")
def test_predict_real_sample_transactions_all_correct(client):
    """Same labeled fixtures used by the C++ (Phase 7) and TensorRT
    (Phase 8) correctness tests."""
    samples = load_sample_transactions()
    assert len(samples) > 0

    n_correct = 0
    for sample in samples:
        response = client.post("/predict", json={"features": sample["features"]})
        assert response.status_code == 200
        body = response.json()
        n_correct += int(body["is_fraud"]) == sample["true_label"]

    assert n_correct == len(samples), f"Only {n_correct}/{len(samples)} real transactions classified correctly"
