"""
Sanity tests for benchmarks/profile_pipeline.py's stage-timing logic.

Skips when CUDA isn't available (same policy as test_gpu_inference.py).
Run with: .venv-gpu\\Scripts\\python.exe -m pytest tests\\test_profiling.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "models" / "checkpoints"
HAS_CHECKPOINT = (CHECKPOINT_DIR / "fraud_mlp.pt").exists()
HAS_CUDA = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not (HAS_CUDA and HAS_CHECKPOINT),
    reason="Requires CUDA (.venv-gpu interpreter) and a trained checkpoint",
)


def test_profile_batch_size_returns_all_stages():
    from benchmarks.profile_pipeline import profile_batch_size
    from src.inference.pytorch_gpu import GPUFraudPredictor

    predictor = GPUFraudPredictor()
    result = profile_batch_size(predictor, batch_size=8)

    expected_stages = {"host_prep", "h2d_transfer", "gpu_compute", "d2h_transfer", "postprocess"}
    assert set(result["mean_ms"].keys()) == expected_stages
    assert set(result["pct_of_total"].keys()) == expected_stages


def test_profile_stage_times_are_positive():
    from benchmarks.profile_pipeline import profile_batch_size
    from src.inference.pytorch_gpu import GPUFraudPredictor

    predictor = GPUFraudPredictor()
    result = profile_batch_size(predictor, batch_size=8)

    for stage, ms in result["mean_ms"].items():
        assert ms > 0, f"{stage} had non-positive mean time"


def test_profile_percentages_sum_to_100():
    from benchmarks.profile_pipeline import profile_batch_size
    from src.inference.pytorch_gpu import GPUFraudPredictor

    predictor = GPUFraudPredictor()
    result = profile_batch_size(predictor, batch_size=8)

    total_pct = sum(result["pct_of_total"].values())
    assert total_pct == pytest.approx(100.0, abs=0.5)
