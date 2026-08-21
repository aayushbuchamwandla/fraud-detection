#!/bin/bash
# Source this inside WSL2 before running anything CUDA/TensorRT-related:
#   source scripts/wsl_env.sh
#
# Sets a clean PATH (avoiding the messy inherited Windows PATH -- see README
# Environment notes) and LD_LIBRARY_PATH pointing at the pip-installed
# cudnn/cuda_runtime/cublas libraries the ~/venv-cuda environment needs
# (see docs/tensorrt.md for why these specific paths are needed).

export PATH="$HOME/venv-cuda/bin:/usr/local/cuda-11.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib"

NVIDIA_PKG_DIR="$HOME/venv-cuda/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_PKG_DIR/cudnn/lib:$NVIDIA_PKG_DIR/cuda_runtime/lib:$NVIDIA_PKG_DIR/cublas/lib"
