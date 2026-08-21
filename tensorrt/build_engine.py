"""
Phase 8, step 2: build a TensorRT engine from the exported ONNX model.

Builds one engine with a dynamic batch dimension, using an optimization
profile that covers every batch size used elsewhere in this project's
benchmarks (1, 32, 128, 512, 1024) -- TensorRT tunes/selects kernels
against the profile's min/opt/max range, so this keeps the engine valid
across the whole benchmark sweep rather than needing five separate engines.

Usage:
    python tensorrt/build_engine.py
"""

from __future__ import annotations

from pathlib import Path

import tensorrt as trt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = PROJECT_ROOT / "models" / "exported"

MIN_BATCH = 1
OPT_BATCH = 256
MAX_BATCH = 1024
INPUT_DIM = 29


def main() -> None:
    onnx_path = EXPORT_DIR / "fraud_model.onnx"
    engine_path = EXPORT_DIR / "fraud_model.engine"

    if not onnx_path.exists():
        raise FileNotFoundError(f"{onnx_path} not found. Run `python tensorrt/export_model.py` first.")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    print(f"Parsed ONNX model: {network.num_layers} layers, "
          f"input '{network.get_input(0).name}' shape {network.get_input(0).shape}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 26)  # 64 MB, plenty for this model

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "features",
        min=(MIN_BATCH, INPUT_DIM),
        opt=(OPT_BATCH, INPUT_DIM),
        max=(MAX_BATCH, INPUT_DIM),
    )
    config.add_optimization_profile(profile)

    print(f"Building engine (min/opt/max batch = {MIN_BATCH}/{OPT_BATCH}/{MAX_BATCH})...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Engine build failed")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    print(f"Saved engine to {engine_path} ({engine_bytes.nbytes:,} bytes)")


if __name__ == "__main__":
    main()
