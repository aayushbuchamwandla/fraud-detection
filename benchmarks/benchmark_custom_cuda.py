"""
Phase 6: custom fused CUDA kernel benchmark.

Same batch sizes, warm-up, and iteration counts as benchmark_cpu.py and
benchmark_gpu.py, so all three are directly comparable. Must be run inside
WSL2 with ~/venv-cuda active (see README Environment notes).

Usage (inside WSL2):
    source ~/venv-cuda/bin/activate
    python benchmarks/benchmark_custom_cuda.py
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
from src.inference.custom_cuda import CustomCUDAPredictor


def main() -> None:
    predictor = CustomCUDAPredictor()
    device = predictor.device

    print(f"Benchmarking custom fused CUDA kernel on {torch.cuda.get_device_name(0)}")
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), threshold={predictor.decision_threshold:.4f}\n")

    def make_input(batch_size: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(42)
        return torch.randn(batch_size, predictor.input_dim, generator=g)

    def predict_fn(x: torch.Tensor):
        return predictor.predict(x)

    results = []
    for batch_size in DEFAULT_BATCH_SIZES:
        latencies = time_inference(predict_fn, make_input, batch_size, device)
        result = summarize(latencies, implementation="custom_cuda_kernel", batch_size=batch_size, device=device)
        results.append(result)
        print(
            f"batch={batch_size:5d} | mean {result.mean_latency_ms:8.4f} ms | "
            f"median {result.median_latency_ms:8.4f} ms | p95 {result.p95_latency_ms:8.4f} ms | "
            f"p99 {result.p99_latency_ms:8.4f} ms | throughput {result.throughput_samples_per_sec:10.1f} samples/s"
        )

    out_path = save_results(results, "custom_cuda_results.json")
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
