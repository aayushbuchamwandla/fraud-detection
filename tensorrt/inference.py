"""
Phase 8, step 3: TensorRT inference -- same predict() interface as every
other backend in this project (CPU, PyTorch GPU, custom CUDA kernel, C++).

Uses pycuda for device memory management. Persistent device buffers are
allocated once for the engine's max batch size (see build_engine.py's
optimization profile) and reused across calls, same design choice as
cpp/inference.py's FraudPredictor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401 -- initializes a CUDA context for this process
import tensorrt as trt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
EXPORT_DIR = PROJECT_ROOT / "models" / "exported"

INPUT_DIM = 29
MAX_BATCH = 1024  # must match build_engine.py's optimization profile max


class TensorRTPredictor:
    def __init__(self, engine_path: Path = EXPORT_DIR / "fraud_model.engine"):
        if not engine_path.exists():
            raise FileNotFoundError(f"{engine_path} not found. Run tensorrt/build_engine.py first.")

        config_path = CHECKPOINT_DIR / "fraud_mlp_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.decision_threshold = json.load(f).get("decision_threshold", 0.5)
        else:
            self.decision_threshold = 0.5

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        # Persistent device buffers sized for the max batch the engine
        # supports -- avoids per-call cudaMalloc, same rationale as
        # cpp/inference.py.
        self.d_input = cuda.mem_alloc(MAX_BATCH * INPUT_DIM * np.dtype(np.float32).itemsize)
        self.d_output = cuda.mem_alloc(MAX_BATCH * np.dtype(np.float32).itemsize)
        self.stream = cuda.Stream()

        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

    def predict_batch_flat(self, x: np.ndarray) -> np.ndarray:
        """x: [batch_size, INPUT_DIM] float32 array. Returns [batch_size] probabilities."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        batch_size = x.shape[0]
        if batch_size > MAX_BATCH:
            raise ValueError(f"batch_size {batch_size} exceeds engine max batch {MAX_BATCH}")

        self.context.set_input_shape(self.input_name, (batch_size, INPUT_DIM))

        cuda.memcpy_htod_async(self.d_input, x, self.stream)
        self.context.execute_async_v3(self.stream.handle)

        out = np.empty(batch_size, dtype=np.float32)
        cuda.memcpy_dtoh_async(out, self.d_output, self.stream)
        self.stream.synchronize()
        return out

    def predict(self, x: np.ndarray) -> dict:
        probs = self.predict_batch_flat(x)
        is_fraud = probs >= self.decision_threshold
        return {"fraud_probability": probs, "is_fraud": is_fraud}

    def predict_single(self, features: list[float]) -> dict:
        x = np.array([features], dtype=np.float32)
        result = self.predict(x)
        return {
            "fraud_probability": float(result["fraud_probability"][0]),
            "is_fraud": bool(result["is_fraud"][0]),
        }


if __name__ == "__main__":
    predictor = TensorRTPredictor()
    print(f"Engine loaded, decision_threshold={predictor.decision_threshold:.4f}")

    dummy = np.random.randn(4, INPUT_DIM).astype(np.float32)
    result = predictor.predict(dummy)
    print("fraud_probability:", result["fraud_probability"])
    print("is_fraud:", result["is_fraud"])
