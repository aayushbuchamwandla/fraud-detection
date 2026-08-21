"""
Benchmark harness shared across every inference backend in this project
(CPU, PyTorch CUDA, custom CUDA, C++, TensorRT) so results are directly
comparable.

Correctness rules this harness follows:
  - Warm-up iterations before timing (JIT/caching/allocator warm-up effects
    would otherwise pollute the first measurements).
  - `torch.cuda.synchronize()` before stopping the timer whenever the tensor
    lives on a CUDA device -- GPU kernel launches are asynchronous from the
    host's perspective, so an un-synchronized timer measures launch overhead,
    not actual compute time.
  - Per-iteration wall-clock samples (not just a single mean) so percentiles
    (P50/P95) are meaningful, not just an average that hides tail latency.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"

DEFAULT_BATCH_SIZES = [1, 32, 128, 512, 1024]
WARMUP_ITERS = 20
MEASURE_ITERS = 200


@dataclass
class BenchmarkResult:
    implementation: str
    batch_size: int
    warmup_iters: int
    measure_iters: int
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_samples_per_sec: float
    device: str
    python_version: str
    torch_version: str
    timestamp: str


def _sync_if_cuda(tensor_or_device) -> None:
    if isinstance(tensor_or_device, torch.device):
        is_cuda = tensor_or_device.type == "cuda"
    else:
        is_cuda = tensor_or_device.is_cuda
    if is_cuda:
        torch.cuda.synchronize()


def time_inference(
    predict_fn,
    make_input_fn,
    batch_size: int,
    device: torch.device,
    warmup_iters: int = WARMUP_ITERS,
    measure_iters: int = MEASURE_ITERS,
) -> list[float]:
    """Time `measure_iters` calls to predict_fn(input), returning per-call
    latencies in milliseconds. `make_input_fn(batch_size)` builds a fresh
    input batch (kept out of the timed region when possible by pre-building
    once and reusing, since input construction is not what we're benchmarking).
    """
    x = make_input_fn(batch_size)

    for _ in range(warmup_iters):
        predict_fn(x)
    _sync_if_cuda(device)

    latencies_ms = []
    for _ in range(measure_iters):
        _sync_if_cuda(device)
        start = time.perf_counter()
        predict_fn(x)
        _sync_if_cuda(device)
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000.0)

    return latencies_ms


def summarize(
    latencies_ms: list[float],
    implementation: str,
    batch_size: int,
    device: torch.device,
    warmup_iters: int = WARMUP_ITERS,
    measure_iters: int = MEASURE_ITERS,
) -> BenchmarkResult:
    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)
    mean_ms = statistics.mean(sorted_lat)
    throughput = batch_size / (mean_ms / 1000.0)

    return BenchmarkResult(
        implementation=implementation,
        batch_size=batch_size,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
        mean_latency_ms=round(mean_ms, 4),
        median_latency_ms=round(statistics.median(sorted_lat), 4),
        p95_latency_ms=round(sorted_lat[int(n * 0.95) - 1], 4),
        p99_latency_ms=round(sorted_lat[int(n * 0.99) - 1], 4),
        throughput_samples_per_sec=round(throughput, 2),
        device=str(device),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def save_results(results: list[BenchmarkResult], filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    return path
