#!/bin/bash
# One-command terminal demo. Run from the project root inside WSL2 for the
# full multi-backend comparison (CPU + PyTorch GPU + custom CUDA + TensorRT);
# run from a plain CPU environment and it gracefully shows CPU-only.
set -e
cd "$(dirname "$0")/.."
source scripts/wsl_env.sh 2>/dev/null || true
python scripts/demo.py
