"""
Phase 4: PyTorch CUDA inference benchmark.

Runs the trained model on the RTX 3060 across the same batch sizes as the
CPU baseline (benchmark_cpu.py) so the two are directly comparable. Must be
run with the .venv-gpu interpreter (torch==1.12.1+cu113) -- see
src/inference/pytorch_gpu.py docstring for why a separate environment.

Usage:
    .venv-gpu\\Scripts\\python.exe benchmarks\\benchmark_gpu.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.benchmark import (
    DEFAULT_BATCH_SIZES,
    save_results,
    summarize,
    time_inference,
)
from src.inference.pytorch_gpu import GPUFraudPredictor


def main() -> None:
    predictor = GPUFraudPredictor()
    device = predictor.device
    n_features = predictor.config.input_dim

    print(f"Benchmarking PyTorch CUDA inference on {torch.cuda.get_device_name(0)}")
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), threshold={predictor.decision_threshold:.4f}\n")

    def make_input(batch_size: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(42)
        return torch.randn(batch_size, n_features, generator=g)

    def predict_fn(x: torch.Tensor):
        return predictor.predict(x)

    results = []
    for batch_size in DEFAULT_BATCH_SIZES:
        latencies = time_inference(predict_fn, make_input, batch_size, device)
        result = summarize(latencies, implementation="pytorch_cuda", batch_size=batch_size, device=device)
        results.append(result)
        print(
            f"batch={batch_size:5d} | mean {result.mean_latency_ms:8.4f} ms | "
            f"median {result.median_latency_ms:8.4f} ms | p95 {result.p95_latency_ms:8.4f} ms | "
            f"p99 {result.p99_latency_ms:8.4f} ms | throughput {result.throughput_samples_per_sec:10.1f} samples/s"
        )

    out_path = save_results(results, "gpu_results.json")
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
