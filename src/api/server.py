"""
FastAPI inference service.

Engine selection: defaults to CPU, not because it's the simplest option but
because every benchmark from Phase 3 onward measured CPU as the FASTEST
backend at batch_size=1 (0.070ms vs 0.091-0.239ms for every GPU path -- see
README/benchmarks/results/), and a REST API's /predict endpoint serves one
transaction at a time. Defaulting to the "most advanced" available engine
would contradict this project's own measurements. Other engines remain
selectable (FRAUD_API_ENGINE env var) for demonstration/benchmarking.

Endpoints:
    GET  /health   -- liveness + which engine is actually loaded
    POST /predict   -- run inference on one transaction's features
"""

from __future__ import annotations

import csv
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_api")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIM = 29
FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
SAMPLES_PATH = PROJECT_ROOT / "data" / "samples" / "sample_transactions.csv"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

VALID_ENGINES = ["cpu", "pytorch_gpu", "custom_cuda", "tensorrt"]


class PredictRequest(BaseModel):
    features: list[float] = Field(..., description=f"Exactly {INPUT_DIM} transaction features (V1-V28, Amount)")

    @field_validator("features")
    @classmethod
    def check_length(cls, v: list[float]) -> list[float]:
        if len(v) != INPUT_DIM:
            raise ValueError(f"Expected {INPUT_DIM} features, got {len(v)}")
        return v


class PredictResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    engine: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    engine: str
    model_loaded: bool
    decision_threshold: float


class SampleTransaction(BaseModel):
    test_split_index: int
    true_label: int
    features: list[float]


def _load_sample_transactions() -> list[SampleTransaction]:
    """Reads the REAL held-out test-split transactions exported by
    scripts/export_sample_transactions.py (see Phase 7) -- the web frontend's
    "Load Legitimate/Fraud Example" buttons call this via GET /samples rather
    than embedding fixture data client-side, so even the example values shown
    are round-tripped through the real backend.
    """
    if not SAMPLES_PATH.exists():
        return []
    with open(SAMPLES_PATH) as f:
        reader = csv.reader(f)
        next(reader)  # header
        return [
            SampleTransaction(
                test_split_index=int(row[0]),
                true_label=int(row[1]),
                features=[float(v) for v in row[2:]],
            )
            for row in reader
        ]


def _load_predictor(engine: str):
    """Load the requested engine's predictor. Raises RuntimeError with a
    clear message if the engine is unavailable in this environment (e.g.
    requesting 'tensorrt' inside a CPU-only container).
    """
    if engine == "cpu":
        from src.inference.cpu_inference import CPUFraudPredictor

        return CPUFraudPredictor(), "cpu"

    if engine == "pytorch_gpu":
        from src.inference.pytorch_gpu import GPUFraudPredictor

        return GPUFraudPredictor(), "pytorch_gpu"

    if engine == "custom_cuda":
        from src.inference.custom_cuda import CustomCUDAPredictor

        return CustomCUDAPredictor(), "custom_cuda"

    if engine == "tensorrt":
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "fraud_trt_inference", PROJECT_ROOT / "tensorrt" / "inference.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.TensorRTPredictor(), "tensorrt"

    raise ValueError(f"Unknown engine {engine!r}. Valid options: {VALID_ENGINES}")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Fraud Detection API",
        description="GPU-accelerated credit card fraud detection inference service",
        version="1.0.0",
    )

    requested_engine = os.environ.get("FRAUD_API_ENGINE", "cpu")
    try:
        predictor, active_engine = _load_predictor(requested_engine)
        logger.info(f"Loaded engine: {active_engine}")
    except Exception as exc:  # noqa: BLE001
        if requested_engine != "cpu":
            logger.warning(f"Failed to load engine {requested_engine!r} ({exc}); falling back to cpu")
            predictor, active_engine = _load_predictor("cpu")
        else:
            raise

    app.state.predictor = predictor
    app.state.engine = active_engine

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            engine=app.state.engine,
            model_loaded=app.state.predictor is not None,
            decision_threshold=app.state.predictor.decision_threshold,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        start = time.perf_counter()
        try:
            result = app.state.predictor.predict_single(request.features)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000.0

        return PredictResponse(
            fraud_probability=result["fraud_probability"],
            is_fraud=result["is_fraud"],
            engine=app.state.engine,
            latency_ms=round(latency_ms, 4),
        )

    @app.get("/samples", response_model=list[SampleTransaction])
    def samples() -> list[SampleTransaction]:
        return _load_sample_transactions()

    @app.get("/feature-names")
    def feature_names() -> dict:
        return {"feature_names": FEATURE_NAMES}

    @app.exception_handler(ValueError)
    def value_error_handler(request, exc: ValueError):  # noqa: ANN001
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    # Serve the web demo frontend (frontend/index.html + assets) as static
    # files at "/". It calls /health, /samples, and /predict via fetch() --
    # no prediction values are computed or hardcoded client-side.
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
