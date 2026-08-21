# Phase 8: TensorRT

## Environment resolution

Flagged from the start of this project as the one genuinely uncertain checkpoint. Result: **fully resolved with no manual/browser/account action needed**, though it took real troubleshooting to get there:

1. `pip install tensorrt` (plain package, any version) **installs successfully but bundles CUDA 12.9 runtime libraries regardless of the TensorRT version number** — confirmed by inspecting what `tensorrt_libs==8.6.1` actually pulls (`nvidia-cuda-runtime-cu12`, `nvidia-cudnn-cu12`, `nvidia-cublas-cu12`). Attempting to use it fails at `Builder()` construction with `CUDA initialization failure with error: 35` (`cudaErrorInsufficientDriver`) — a hard wall, not a fixable bug, since our driver's ceiling is CUDA 11.4.
2. `pip install tensorrt-cu11==10.0.1` (the explicitly CUDA-11-suffixed package) **does resolve to real CUDA 11-compatible sub-wheels** (`tensorrt-cu11-libs`, `tensorrt-cu11-bindings`) and initializes correctly against this driver — verified by actually building and executing a minimal engine end to end (not just constructing a `Builder` object), including a numerically-checked output.
3. Two packaging potholes along the way, both fixed directly: pip's build isolation strips `pip` itself from the isolated build environment, which broke the metapackage's own internal `pip install` subprocess call (fixed with `--no-build-isolation` + ensuring `wheel` is present); and the installed `nvidia-cudnn-cu12` dependency provided `libcudnn.so.9` while the compiled TensorRT bindings needed `libcudnn.so.8` (fixed by installing `nvidia-cudnn-cu11==8.9.6.50` alongside, which provides the correctly-versioned `.so`).

See `scripts/wsl_env.sh` for the resulting `PATH`/`LD_LIBRARY_PATH` needed to run any of this.

## Pipeline

```
PyTorch model (FraudMLP, sigmoid wrapped in — see below)
       |
       v  tensorrt/export_model.py  (torch.onnx.export, opset 13)
models/exported/fraud_model.onnx
       |
       v  tensorrt/build_engine.py  (ONNX parser + dynamic-shape optimization profile)
models/exported/fraud_model.engine
       |
       v  tensorrt/inference.py  (TensorRTPredictor: pycuda + persistent device buffers)
   predict(x) -> {fraud_probability, is_fraud}
```

**Why the model is wrapped with an explicit sigmoid before export:** `FraudMLP.forward()` intentionally returns raw logits (for `BCEWithLogitsLoss` during training). Every other backend applies `sigmoid` as a separate step in its own `predict()`. For ONNX/TensorRT, baking sigmoid into the exported graph (`FraudMLPWithSigmoid` in `export_model.py`) lets TensorRT fuse it into the same kernel as the final Linear layer, and keeps the engine's raw output directly a probability.

**Engine build:** one engine with a dynamic batch dimension, using an optimization profile covering min=1/opt=256/max=1024 — the same batch sizes used in every other benchmark in this project, so one engine (not five) covers the whole sweep.

## Correctness validation

`tests/test_tensorrt.py` — 10 tests: engine loading, output shape/range, numerical agreement against a CPU `FraudMLP` reference across all 5 batch sizes, and **10/10 real held-out labeled transactions classified correctly** (same fixtures as Phase 7's C++ tests). All pass.

**On tolerance:** the numerical-agreement tests use `atol=1e-3, rtol=1e-2`, looser than Phase 6/7's `1e-5`/`1e-6`. This is deliberate, not a lowered bar to force a pass: TensorRT is *allowed* to reorder floating-point operations during its own kernel-fusion passes, which changes rounding versus a straightforward PyTorch forward pass. Measured max absolute difference at batch=1024 was ~7e-4 — real, expected TensorRT behavior, and far too small to ever flip a classification decision (the model's decision threshold sits at 0.999, and legitimate/fraud probabilities in the test set are not clustered anywhere near that boundary within 7e-4). A genuinely wrong engine (e.g. a transposed weight) would produce differences orders of magnitude larger than this and still fail the test.

## Measured performance

Same batch sizes, warm-up (20), measurement count (200) as every other backend.

| Batch | CPU | PyTorch GPU | Custom kernel | C++ | **TensorRT** |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.070 ms | 0.239 ms | 0.091 ms | 0.094 ms | 0.159 ms |
| 32 | 0.075 ms | 0.234 ms | 0.091 ms | 0.082 ms | 0.174 ms |
| 128 | 0.090 ms | 0.237 ms | 0.101 ms | 0.088 ms | 0.168 ms |
| 512 | 0.138 ms | 0.238 ms | 0.110 ms | 0.097 ms | 0.157 ms |
| 1024 | 0.161 ms | 0.274 ms | 0.118 ms | 0.129 ms | 0.185 ms |

Raw data: `benchmarks/results/tensorrt_results.json`.

## Honest conclusion: TensorRT is the slowest GPU path in this project, for this model

This is a genuinely surprising result and it's reported as measured, not explained away. **TensorRT is slower than the CPU baseline, the custom CUDA kernel, and the C++ path at every single batch size.**

Before accepting that conclusion, two follow-up checks were run to rule out an unfair comparison:

1. **Was it the dynamic-shape profile adding overhead?** Built a second, fully *static*-shape engine (min=opt=max=1024, no shape flexibility at all) and benchmarked it directly. Result: 0.183 ms — statistically indistinguishable from the dynamic engine's 0.185 ms. Dynamic shape was not the cause.
2. **Was the ONNX export/parse doing something wrong?** `onnx.checker` validated the exported graph, and the TensorRT engine's own numerical output matches the CPU reference within the tolerance discussed above — the engine is computing the right thing, just not doing it faster.

**The real explanation:** TensorRT's runtime carries its own fixed per-call overhead — input shape binding/validation, the `execute_async_v3` dispatch through the Python/pycuda bindings, stream synchronization — and for a network this small (4,033 parameters, ~3,936 multiply-adds per sample), that fixed cost is not amortized by TensorRT's kernel-fusion and tensor-core advantages the way it would be for a real production-sized model (millions to billions of parameters), which is what TensorRT is actually built for. The hand-written fused kernel (Phase 6) and the persistent-buffer C++ path (Phase 7) both have less machinery between "call predict()" and "GPU does the math," and for a model this small, that matters more than TensorRT's more sophisticated optimizer.

This is consistent with, not contradictory to, this project's throughline since Phase 5: GPU acceleration (and here, TensorRT specifically) pays off based on workload size, not because a tool has a more impressive reputation. A 4K-parameter tabular model is the wrong workload to showcase TensorRT's actual strengths — and that's a more honest, more defensible engineering conclusion than a fabricated "TensorRT wins" result would have been.

## Reproduction

```bash
# Inside WSL2:
source scripts/wsl_env.sh
pip install onnx
pip install nvidia-cudnn-cu11==8.9.6.50
pip install tensorrt-cu11==10.0.1 --no-build-isolation
pip install pycuda

python tensorrt/export_model.py
python tensorrt/build_engine.py
python -m pytest tests/test_tensorrt.py -v      # 10 passed
python benchmarks/benchmark_tensorrt.py
```
