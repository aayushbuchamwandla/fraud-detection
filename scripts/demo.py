"""
Terminal demo: loads the trained model and labeled sample transactions,
then runs inference through every backend available in the current
environment. Which backends run depends on what's importable right now
(see scripts/check_environment.py for the same detection logic).

For each transaction, every available backend is printed ranked
fastest-to-slowest with a latency bar, followed by an aggregate
mean-latency summary across all transactions.

Usage:
    python scripts/demo.py
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SAMPLES_PATH = PROJECT_ROOT / "data" / "samples" / "sample_transactions.csv"


def load_samples() -> list[dict]:
    with open(SAMPLES_PATH) as f:
        reader = csv.reader(f)
        next(reader)
        rows = [{"idx": int(r[0]), "true_label": int(r[1]), "features": [float(v) for v in r[2:]]} for r in reader]
    # 2 legitimate + 2 fraud, for a balanced demo
    legit = [r for r in rows if r["true_label"] == 0][:2]
    fraud = [r for r in rows if r["true_label"] == 1][:2]
    return legit + fraud


def get_backends() -> dict:
    """Returns {name: predictor_instance} for every backend importable in
    this environment right now."""
    backends = {}

    from src.inference.cpu_inference import CPUFraudPredictor

    backends["cpu"] = CPUFraudPredictor()

    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            try:
                from src.inference.pytorch_gpu import GPUFraudPredictor

                backends["pytorch_gpu"] = GPUFraudPredictor()
            except Exception as exc:  # noqa: BLE001
                print(f"(pytorch_gpu unavailable: {exc})")

            try:
                from src.inference.custom_cuda import CustomCUDAPredictor

                backends["custom_cuda"] = CustomCUDAPredictor()
            except Exception as exc:  # noqa: BLE001
                print(f"(custom_cuda unavailable: {exc})")

    engine_path = PROJECT_ROOT / "models" / "exported" / "fraud_model.engine"
    from scripts.check_environment import check_tensorrt

    if check_tensorrt()["available"] and engine_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("fraud_trt_inference", PROJECT_ROOT / "tensorrt" / "inference.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            backends["tensorrt"] = module.TensorRTPredictor()
        except Exception as exc:  # noqa: BLE001
            print(f"(tensorrt unavailable: {exc})")

    return backends


def text_bar(value: float, max_value: float, width: int = 30) -> str:
    """Renders a value as a filled Unicode block bar, scaled against
    max_value -- makes the latency spread visible at a glance on a terminal
    recording without needing an external chart."""
    filled = max(1, round(width * value / max_value)) if max_value > 0 else 0
    return "█" * filled + "░" * (width - filled)


def predict_one(name: str, predictor, features: list[float]) -> tuple[float, bool, float]:
    start = time.perf_counter()
    if name == "tensorrt":
        import numpy as np

        result = predictor.predict(np.array([features], dtype="float32"))
        prob, is_fraud = float(result["fraud_probability"][0]), bool(result["is_fraud"][0])
    else:
        result = predictor.predict_single(features)
        prob, is_fraud = result["fraud_probability"], result["is_fraud"]
    latency_ms = (time.perf_counter() - start) * 1000.0
    return prob, is_fraud, latency_ms


def main() -> None:
    print("=" * 60)
    print("GPU FRAUD DETECTION DEMO")
    print("=" * 60)

    from scripts.check_environment import check_gpu, check_torch

    gpu = check_gpu()
    if gpu["available"]:
        print(f"\nGPU: {gpu['name']}")
        print(f"Driver: {gpu['driver_version']}")
    else:
        print("\nGPU: not detected in this environment")

    torch_info = check_torch()
    if torch_info["available"] and torch_info.get("cuda_available"):
        print(f"CUDA: {torch_info['cuda_version']} (via torch {torch_info['version']})")

    print("\nLoading trained model + real labeled transactions...")
    samples = load_samples()
    backends = get_backends()
    print(f"Inference engines available in this environment: {list(backends.keys())}")

    # One warm-up call per backend before timing anything -- the first GPU
    # call pays one-time CUDA context / cuDNN algorithm-search cost
    # (measured here: ~4.6s cold-start for pytorch_gpu), which would
    # otherwise dominate the first transaction's latency number.
    warmup_sample = samples[0]["features"]
    for name, predictor in backends.items():
        predict_one(name, predictor, warmup_sample)

    all_latencies: dict[str, list[float]] = {name: [] for name in backends}

    for sample in samples:
        label_str = "FRAUD" if sample["true_label"] else "LEGITIMATE"
        print(f"\nTransaction (test-split index {sample['idx']}, true label: {label_str})")

        per_backend_results = []
        for name, predictor in backends.items():
            prob, is_fraud, latency_ms = predict_one(name, predictor, sample["features"])
            per_backend_results.append((name, prob, is_fraud, latency_ms))
            all_latencies[name].append(latency_ms)

        # Rank fastest -> slowest for this transaction, with a bar chart
        # showing the latency spread across backends.
        max_latency = max(r[3] for r in per_backend_results)
        for rank, (name, prob, is_fraud, latency_ms) in enumerate(
            sorted(per_backend_results, key=lambda r: r[3]), start=1
        ):
            pred_str = "FRAUD" if is_fraud else "LEGITIMATE"
            correct = "correct" if is_fraud == bool(sample["true_label"]) else "WRONG"
            bar = text_bar(latency_ms, max_latency)
            print(
                f"  {rank}. [{name:12s}] {bar} {latency_ms:7.4f} ms  "
                f"{pred_str:12s} p={prob:.4f}  ({correct})"
            )

    # Mean across all transactions is less noisy than any single result.
    print("\n" + "=" * 60)
    print("SUMMARY: mean latency per backend, across all transactions above")
    print("=" * 60)
    means = {name: sum(lats) / len(lats) for name, lats in all_latencies.items() if lats}
    max_mean = max(means.values())
    for rank, (name, mean_ms) in enumerate(sorted(means.items(), key=lambda kv: kv[1]), start=1):
        bar = text_bar(mean_ms, max_mean)
        print(f"  {rank}. {name:12s} {bar} {mean_ms:7.4f} ms")

    print(
        "\nNote: these are single live calls (one warm-up + one measured call per\n"
        "transaction), not the rigorous 200-iteration benchmark -- the relative\n"
        "order between close backends (e.g. custom CUDA vs TensorRT) can differ\n"
        "run to run here even though it's stable in the formal benchmark. See\n"
        "benchmarks/results/ and the README's benchmark tables for the\n"
        "statistically-controlled comparison this project's conclusions are\n"
        "actually based on."
    )

    print("\n" + "=" * 60)
    print("Demo complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
