"""
Full project verification: actually runs each major component and reports
[PASS]/[FAIL]/[SKIP] based on what genuinely happened -- not a checklist
that gets marked done by inspection. [SKIP] means the component's
dependency isn't available in this environment (e.g. TensorRT on the plain
Windows CPU venv), which is a legitimate, honestly-reported outcome, not a
failure. [FAIL] means something that should have worked didn't.

Usage:
    python scripts/verify.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

results: list[tuple[str, str, str]] = []  # (name, status, detail)


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    marker = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[status]
    print(f"{marker} {name}" + (f" -- {detail}" if detail else ""))


def check_python_env() -> None:
    try:
        import torch  # noqa: F401

        record("Python environment", "PASS", f"torch importable, {sys.version.split()[0]}")
    except ImportError as exc:
        record("Python environment", "FAIL", str(exc))


def check_dataset() -> None:
    raw = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
    if raw.exists():
        record("Dataset (raw)", "PASS", f"{raw}")
    else:
        record("Dataset (raw)", "SKIP", "run scripts/fetch_dataset.py")

    processed = PROJECT_ROOT / "data" / "processed" / "X_train.npy"
    if processed.exists():
        record("Dataset (processed)", "PASS", f"{processed}")
    else:
        record("Dataset (processed)", "SKIP", "run python -m src.data.preprocessing")


def check_model() -> None:
    ckpt = PROJECT_ROOT / "models" / "checkpoints" / "fraud_mlp.pt"
    if not ckpt.exists():
        record("Trained model", "SKIP", "run python -m src.models.train")
        return
    try:
        import torch

        from src.models.model import FraudMLP, ModelConfig

        model = FraudMLP(ModelConfig(input_dim=29))
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
        out = model.predict_proba(torch.randn(1, 29))
        assert 0.0 <= out.item() <= 1.0
        record("Trained model", "PASS", f"loads + produces valid probability ({ckpt})")
    except Exception as exc:  # noqa: BLE001
        record("Trained model", "FAIL", str(exc))


def check_gpu() -> None:
    from scripts.check_environment import check_gpu as _check_gpu
    from scripts.check_environment import check_torch as _check_torch

    gpu = _check_gpu()
    record("GPU detected (nvidia-smi)", "PASS" if gpu["available"] else "SKIP", gpu.get("name", "not found"))

    torch_info = _check_torch()
    if torch_info["available"] and torch_info.get("cuda_available"):
        record("CUDA available to PyTorch", "PASS", f"torch {torch_info['version']}")
    else:
        record("CUDA available to PyTorch", "SKIP", "no CUDA-enabled torch in this environment")


def check_cpu_inference() -> None:
    ckpt = PROJECT_ROOT / "models" / "checkpoints" / "fraud_mlp.pt"
    if not ckpt.exists():
        record("CPU inference", "SKIP", "no trained model")
        return
    try:
        import torch

        from src.inference.cpu_inference import CPUFraudPredictor

        predictor = CPUFraudPredictor()
        result = predictor.predict(torch.randn(4, 29))
        assert result["fraud_probability"].shape == (4,)
        record("CPU inference", "PASS")
    except Exception as exc:  # noqa: BLE001
        record("CPU inference", "FAIL", str(exc))


def check_gpu_inference() -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            record("PyTorch GPU inference", "SKIP", "no CUDA-enabled torch in this environment")
            return
        from src.inference.pytorch_gpu import GPUFraudPredictor

        predictor = GPUFraudPredictor()
        result = predictor.predict(torch.randn(4, 29))
        assert result["fraud_probability"].shape == (4,)
        record("PyTorch GPU inference", "PASS")
    except Exception as exc:  # noqa: BLE001
        record("PyTorch GPU inference", "FAIL", str(exc))


def check_custom_cuda() -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            record("Custom CUDA kernel", "SKIP", "no CUDA-enabled torch in this environment")
            return
        from src.inference.custom_cuda import CustomCUDAPredictor

        predictor = CustomCUDAPredictor()
        result = predictor.predict(torch.randn(4, 29))
        assert result["fraud_probability"].shape == (4,)
        record("Custom CUDA kernel", "PASS")
    except Exception as exc:  # noqa: BLE001
        record("Custom CUDA kernel", "FAIL", str(exc))


def check_tensorrt() -> None:
    from scripts.check_environment import check_tensorrt as _check_trt

    if not _check_trt()["available"]:
        record("TensorRT", "SKIP", "tensorrt not installed in this environment")
        return
    engine_path = PROJECT_ROOT / "models" / "exported" / "fraud_model.engine"
    if not engine_path.exists():
        record("TensorRT", "SKIP", "no engine built -- run tensorrt/build_engine.py")
        return
    try:
        import importlib.util

        import numpy as np

        spec = importlib.util.spec_from_file_location("fraud_trt_inference", PROJECT_ROOT / "tensorrt" / "inference.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        predictor = module.TensorRTPredictor()
        result = predictor.predict(np.random.randn(4, 29).astype(np.float32))
        assert result["fraud_probability"].shape == (4,)
        record("TensorRT", "PASS")
    except Exception as exc:  # noqa: BLE001
        record("TensorRT", "FAIL", str(exc))


def check_api() -> None:
    try:
        from fastapi.testclient import TestClient

        from src.api.server import create_app

        client = TestClient(create_app())
        health = client.get("/health")
        predict = client.post("/predict", json={"features": [0.0] * 29})
        assert health.status_code == 200 and predict.status_code == 200
        record("REST API", "PASS", "GET /health and POST /predict both succeeded")
    except Exception as exc:  # noqa: BLE001
        record("REST API", "FAIL", str(exc))


def check_docker() -> None:
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            record("Docker", "SKIP", "docker not available in this environment")
            return
        record("Docker CLI", "PASS", result.stdout.strip())
    except FileNotFoundError:
        record("Docker", "SKIP", "docker not available in this environment")


def check_tests() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"], capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    last_line = [l for l in result.stdout.strip().splitlines() if l][-1] if result.stdout.strip() else ""
    if result.returncode == 0:
        record("Automated test suite", "PASS", last_line)
    else:
        record("Automated test suite", "FAIL", last_line)


def main() -> None:
    print("=" * 60)
    print("PROJECT VERIFICATION")
    print("=" * 60)

    check_python_env()
    check_dataset()
    check_model()
    check_gpu()
    check_cpu_inference()
    check_gpu_inference()
    check_custom_cuda()
    check_tensorrt()
    check_api()
    check_docker()
    check_tests()

    print("\n" + "=" * 60)
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    n_skip = sum(1 for _, s, _ in results if s == "SKIP")
    print(f"Verification complete: {n_pass} passed, {n_skip} skipped, {n_fail} failed")
    print("=" * 60)

    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
