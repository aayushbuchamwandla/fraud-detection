"""
Phase 3: CPU inference baseline benchmark.

Runs the trained model on CPU across several batch sizes and records
mean/median/P95/P99 latency and throughput. This is the baseline every
later phase (PyTorch GPU, custom CUDA, C++, TensorRT) is compared against.

Usage:
    python benchmarks/benchmark_cpu.py
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
from src.inference.cpu_inference import CPUFraudPredictor


def main() -> None:
    predictor = CPUFraudPredictor()
    device = predictor.device
    n_features = predictor.config.input_dim

    print(f"Benchmarking CPU inference (device={device}, threads={torch.get_num_threads()})")
    print(f"Model: {n_features} input features, threshold={predictor.decision_threshold:.4f}\n")

    def make_input(batch_size: int) -> torch.Tensor:
        # Fixed seed per call for reproducible input shape/values across runs
        g = torch.Generator().manual_seed(42)
        return torch.randn(batch_size, n_features, generator=g)

    def predict_fn(x: torch.Tensor):
        return predictor.predict(x)

    results = []
    for batch_size in DEFAULT_BATCH_SIZES:
        latencies = time_inference(predict_fn, make_input, batch_size, device)
        result = summarize(latencies, implementation="cpu_pytorch", batch_size=batch_size, device=device)
        results.append(result)
        print(
            f"batch={batch_size:5d} | mean {result.mean_latency_ms:8.4f} ms | "
            f"median {result.median_latency_ms:8.4f} ms | p95 {result.p95_latency_ms:8.4f} ms | "
            f"p99 {result.p99_latency_ms:8.4f} ms | throughput {result.throughput_samples_per_sec:10.1f} samples/s"
        )

    out_path = save_results(results, "cpu_results.json")
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
