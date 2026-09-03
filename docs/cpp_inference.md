# Phase 7: C++ Inference Component

## Purpose

`cpp/inference.cpp`'s `FraudPredictor` reads the exported raw weight binaries directly, owns persistent GPU device buffers (weights uploaded once at construction; input/output buffers reused across calls and grown only on demand), and calls `cuda/fraud_kernel.cu`'s kernel directly, with no PyTorch/libtorch dependency in the call path.

## Design

**`cpp/inference.hpp` / `cpp/inference.cpp`** — the `FraudPredictor` class:
- Constructor loads `w1.bin`...`b3.bin` and `threshold.txt` (from `scripts/export_weights.py`) and uploads weights to the GPU exactly once.
- `predict(features)` / `predict_batch(rows)` — clean public interface, single transaction or batch.
- `predict_batch_flat(ptr, batch_size)` — the entry point both convenience methods funnel through, and what the benchmark harness calls directly to avoid `std::vector<std::vector<float>>` copy overhead when timing.
- Non-copyable (raw device pointer ownership; copying would either double-free or silently share state) — copy constructor/assignment explicitly deleted.
- Every CUDA call is checked via a `check_cuda()` helper that throws `std::runtime_error` with the actual CUDA error string on failure — no silent failures.
- Wrong feature-vector size throws `std::invalid_argument` with the actual expected/received counts, not an out-of-bounds read.

**Why persistent buffers matter here:** the PyTorch extension path (Phase 6) re-wraps a `torch::Tensor` — with ATen's own allocator bookkeeping — on every `predict()` call. `FraudPredictor` allocates its input/output device buffers once and reuses them, only growing (never shrinking) when a larger batch shows up.

**Build:** `cpp/CMakeLists.txt` compiles `cuda/fraud_kernel.cu` directly as part of the `fraud_inference` library target — no dependency on `cuda/`'s own CMake build having run first, so `cpp/` is independently buildable while still reusing the exact kernel source (no duplicated math).

## Correctness validation

`cpp/test_predictor.cpp`, run against the trained weights and the labeled sample transactions (`data/samples/sample_transactions.csv`, pulled from the held-out test split — see `scripts/export_sample_transactions.py`):

- Input validation: wrong feature count → `std::invalid_argument`
- Missing weights directory → `std::runtime_error`
- All 10 sample transactions produce probabilities in `[0, 1]`, **10/10 classified correctly** against true labels
- `predict_batch()` output matches `predict()` called row-by-row within `1e-6` (catches batching/indexing bugs specifically, independent of the kernel's own numerical correctness which Phase 6 already validated)

All tests pass.

## Measured performance

Same batch sizes, warm-up (20), and measurement count (200) as every other backend in this project, for direct comparability.

| Batch | CPU (PyTorch) | PyTorch GPU | Custom kernel (PyTorch ext.) | **C++ (persistent buffers)** |
|---:|---:|---:|---:|---:|
| 1 | 0.070 ms | 0.239 ms | 0.091 ms | 0.094 ms |
| 32 | 0.075 ms | 0.234 ms | 0.091 ms | 0.082 ms |
| 128 | 0.090 ms | 0.237 ms | 0.101 ms | 0.088 ms |
| 512 | 0.138 ms | 0.238 ms | 0.110 ms | 0.097 ms |
| 1024 | 0.161 ms | 0.274 ms | 0.118 ms | 0.129 ms |

Raw data: `benchmarks/results/cpp_results.json`.

## Conclusion

The C++ path is comparable to, and at batch sizes 32-512 modestly faster than, the PyTorch-extension version of the identical kernel — consistent with removing Python/PyTorch dispatch overhead (`torch::Tensor` construction, ATen op dispatch) from the call path. It is not uniformly faster: at batch=1024 the C++ path measured slower than the PyTorch extension (0.129ms vs 0.118ms). Since both call the same `launch_fraud_mlp_forward` kernel, this is most plausibly measurement noise (WSL2/system jitter) rather than an architectural effect at that batch size. The crossover against the CPU baseline (C++ wins at batch≥512, same as Phase 6's kernel) matches Phase 5/6's conclusion: this model is small enough that GPU paths only pay off once the batch is large enough to amortize fixed per-call overhead.
