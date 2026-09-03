"""
Phase 5: stage-by-stage profiling of the GPU inference pipeline.

Phase 4 found the GPU slower than CPU at every batch size for this model.
This script times each pipeline stage separately to identify where the time
actually goes before optimizing anything:

    host tensor prep -> H2D transfer -> GPU compute -> D2H transfer -> postprocess

Must be run with the .venv-gpu interpreter (torch==1.12.1+cu113):
    .venv-gpu\\Scripts\\python.exe benchmarks\\profile_pipeline.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.pytorch_gpu import GPUFraudPredictor

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BATCH_SIZES = [1, 32, 128, 512, 1024]
WARMUP_ITERS = 20
MEASURE_ITERS = 200


def profile_batch_size(predictor: GPUFraudPredictor, batch_size: int) -> dict:
    device = predictor.device
    n_features = predictor.config.input_dim
    threshold = predictor.decision_threshold

    g = torch.Generator().manual_seed(42)
    x_cpu = torch.randn(batch_size, n_features, generator=g)

    # Warm-up: exercises CUDA context/allocator/cuDNN autotune so those
    # one-time costs don't pollute the measured stages.
    for _ in range(WARMUP_ITERS):
        x_gpu = x_cpu.to(device)
        with torch.no_grad():
            logits = predictor.model(x_gpu)
            probs = torch.sigmoid(logits)
            _ = probs >= threshold
        torch.cuda.synchronize()

    stage_times_ms = {"host_prep": [], "h2d_transfer": [], "gpu_compute": [], "d2h_transfer": [], "postprocess": []}

    for _ in range(MEASURE_ITERS):
        # Stage 1: host tensor prep (simulates receiving a batch of features)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x_cpu_iter = x_cpu.clone()
        t1 = time.perf_counter()

        # Stage 2: host -> device transfer
        x_gpu = x_cpu_iter.to(device)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        # Stage 3: GPU compute (forward pass + sigmoid)
        with torch.no_grad():
            logits = predictor.model(x_gpu)
            probs_gpu = torch.sigmoid(logits)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        # Stage 4: device -> host transfer
        probs_cpu = probs_gpu.cpu()
        t4 = time.perf_counter()

        # Stage 5: postprocess (threshold comparison, on CPU where the caller needs it)
        _ = probs_cpu >= threshold
        t5 = time.perf_counter()

        stage_times_ms["host_prep"].append((t1 - t0) * 1000)
        stage_times_ms["h2d_transfer"].append((t2 - t1) * 1000)
        stage_times_ms["gpu_compute"].append((t3 - t2) * 1000)
        stage_times_ms["d2h_transfer"].append((t4 - t3) * 1000)
        stage_times_ms["postprocess"].append((t5 - t4) * 1000)

    def mean(vals: list[float]) -> float:
        return sum(vals) / len(vals)

    means = {stage: round(mean(vals), 5) for stage, vals in stage_times_ms.items()}
    total = sum(means.values())
    pct = {stage: round(100 * v / total, 1) for stage, v in means.items()}

    return {
        "batch_size": batch_size,
        "mean_ms": means,
        "pct_of_total": pct,
        "total_mean_ms": round(total, 5),
    }


def main() -> None:
    predictor = GPUFraudPredictor()
    print(f"Profiling pipeline on {torch.cuda.get_device_name(0)}")
    print(f"Model: {predictor.config.input_dim} features, {WARMUP_ITERS} warmup + {MEASURE_ITERS} measured iters\n")

    results = []
    for batch_size in BATCH_SIZES:
        r = profile_batch_size(predictor, batch_size)
        results.append(r)
        print(f"--- batch_size={batch_size} (total {r['total_mean_ms']:.4f} ms) ---")
        for stage, ms in r["mean_ms"].items():
            print(f"  {stage:15s} {ms:9.5f} ms  ({r['pct_of_total'][stage]:5.1f}%)")
        print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "pipeline_profile.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
