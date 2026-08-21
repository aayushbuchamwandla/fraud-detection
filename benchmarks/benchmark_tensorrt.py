"""
Phase 8: TensorRT inference benchmark.

Same batch sizes, warm-up (20), and measurement count (200) as every other
backend in this project. TensorRT/pycuda don't use torch tensors, so this
uses its own timing loop (pycuda Stream.synchronize() instead of
torch.cuda.synchronize()) but reuses src/inference/benchmark.py's
summarize()/save_results() for an identical output schema.

Usage (inside WSL2):
    source scripts/wsl_env.sh
    python benchmarks/benchmark_tensorrt.py
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.benchmark import DEFAULT_BATCH_SIZES, save_results, summarize  # noqa: E402

WARMUP_ITERS = 20
MEASURE_ITERS = 200


def _load_tensorrt_predictor_class():
    # Loaded by explicit file path -- see tests/test_tensorrt.py for why
    # (tensorrt/ shares its name with the real pip `tensorrt` package).
    spec = importlib.util.spec_from_file_location("fraud_trt_inference", PROJECT_ROOT / "tensorrt" / "inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TensorRTPredictor


def main() -> None:
    TensorRTPredictor = _load_tensorrt_predictor_class()
    predictor = TensorRTPredictor()

    print(f"Benchmarking TensorRT inference, decision_threshold={predictor.decision_threshold:.4f}\n")

    results = []
    for batch_size in DEFAULT_BATCH_SIZES:
        rng = np.random.default_rng(42)
        x = rng.standard_normal((batch_size, 29)).astype(np.float32)

        for _ in range(WARMUP_ITERS):
            predictor.predict(x)

        latencies_ms = []
        for _ in range(MEASURE_ITERS):
            start = time.perf_counter()
            predictor.predict(x)  # predict() itself calls stream.synchronize()
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)

        result = summarize(latencies_ms, implementation="tensorrt", batch_size=batch_size, device="cuda (TensorRT)")
        results.append(result)
        print(
            f"batch={batch_size:5d} | mean {result.mean_latency_ms:8.4f} ms | "
            f"median {result.median_latency_ms:8.4f} ms | p95 {result.p95_latency_ms:8.4f} ms | "
            f"p99 {result.p99_latency_ms:8.4f} ms | throughput {result.throughput_samples_per_sec:10.1f} samples/s"
        )

    out_path = save_results(results, "tensorrt_results.json")
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
