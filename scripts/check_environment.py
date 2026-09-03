"""
Reports GPU/CUDA/TensorRT/compiler/Docker availability in the current
environment via subprocess calls and imports. Output differs across this
project's three environments (.venv, .venv-gpu, WSL2 ~/venv-cuda).

Usage:
    python scripts/check_environment.py
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys


def run(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def check_gpu() -> dict:
    output = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    if output is None:
        return {"available": False}
    parts = [p.strip() for p in output.split(",")]
    return {
        "available": True,
        "name": parts[0] if len(parts) > 0 else "unknown",
        "memory": parts[1] if len(parts) > 1 else "unknown",
        "driver_version": parts[2] if len(parts) > 2 else "unknown",
    }


def check_torch() -> dict:
    if importlib.util.find_spec("torch") is None:
        return {"available": False}
    import torch

    info = {"available": True, "version": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["device_name"] = torch.cuda.get_device_name(0)
    return info


def check_tensorrt() -> dict:
    # This project has its own tensorrt/ directory at the repo root, which
    # Python's implicit namespace-package mechanism (PEP 420) will import
    # even when the real NVIDIA tensorrt package isn't installed --
    # find_spec() alone returns a truthy spec either way. tensorrt.__file__
    # distinguishes them: the real package has one, a namespace package's
    # is None.
    try:
        import tensorrt as trt
    except ImportError:
        return {"available": False}

    if getattr(trt, "__file__", None) is None or not hasattr(trt, "__version__"):
        return {"available": False}
    return {"available": True, "version": trt.__version__}


def check_nvcc() -> dict:
    output = run(["nvcc", "--version"])
    if output is None:
        return {"available": False}
    for line in output.splitlines():
        if "release" in line.lower():
            return {"available": True, "version_line": line.strip()}
    return {"available": True, "version_line": output.splitlines()[-1]}


def check_cmake() -> dict:
    output = run(["cmake", "--version"])
    if output is None:
        return {"available": False}
    return {"available": True, "version_line": output.splitlines()[0]}


def check_docker() -> dict:
    output = run(["docker", "--version"])
    if output is None:
        return {"available": False}
    return {"available": True, "version_line": output.strip()}


def check_gcc() -> dict:
    for compiler in ["g++-10", "g++", "cl"]:
        output = run([compiler, "--version"])
        if output is not None:
            return {"available": True, "compiler": compiler, "version_line": output.splitlines()[0]}
    return {"available": False}


def fmt(available: bool) -> str:
    return "AVAILABLE" if available else "UNAVAILABLE"


def main() -> None:
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)

    print(f"\nOS: {platform.system()} {platform.release()} ({platform.platform()})")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")

    gpu = check_gpu()
    print(f"\nGPU: {fmt(gpu['available'])}")
    if gpu["available"]:
        print(f"  Name: {gpu['name']}")
        print(f"  VRAM: {gpu['memory']}")
        print(f"  Driver: {gpu['driver_version']}")

    torch_info = check_torch()
    print(f"\nPyTorch: {fmt(torch_info['available'])}")
    if torch_info["available"]:
        print(f"  Version: {torch_info['version']}")
        print(f"  CUDA available: {torch_info['cuda_available']}")
        if torch_info["cuda_available"]:
            print(f"  CUDA version (torch built for): {torch_info['cuda_version']}")
            print(f"  Device: {torch_info['device_name']}")

    trt_info = check_tensorrt()
    print(f"\nTensorRT: {fmt(trt_info['available'])}")
    if trt_info["available"]:
        print(f"  Version: {trt_info['version']}")

    nvcc_info = check_nvcc()
    print(f"\nnvcc (CUDA compiler): {fmt(nvcc_info['available'])}")
    if nvcc_info["available"]:
        print(f"  {nvcc_info['version_line']}")

    gcc_info = check_gcc()
    print(f"\nC++ compiler: {fmt(gcc_info['available'])}")
    if gcc_info["available"]:
        print(f"  {gcc_info['compiler']}: {gcc_info['version_line']}")

    cmake_info = check_cmake()
    print(f"\nCMake: {fmt(cmake_info['available'])}")
    if cmake_info["available"]:
        print(f"  {cmake_info['version_line']}")

    docker_info = check_docker()
    print(f"\nDocker: {fmt(docker_info['available'])}")
    if docker_info["available"]:
        print(f"  {docker_info['version_line']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
