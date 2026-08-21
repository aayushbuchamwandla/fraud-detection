# Phase 6: Custom CUDA Kernel

## Purpose

Phase 5 measured that PyTorch's eager-mode dispatch for `FraudMLP.forward()` spends ~70-73% of total pipeline time in `gpu_compute`, and that time is nearly flat across batch sizes 1-1024 — the signature of fixed per-kernel-launch overhead (~6 launches: 3x Linear/GEMM, 2x ReLU, 1x sigmoid) rather than compute scaling with data. This kernel targets exactly that: fuse the entire 3-layer forward pass into **one** kernel launch.

## Design

**File:** `cuda/fraud_kernel.cu` / `cuda/fraud_kernel.cuh`

**Granularity: one CUDA thread per transaction**, not one thread block per sample or one thread per output neuron. Each thread independently runs the full forward pass for its row — 29→64→32→1, roughly 3,936 multiply-adds total. This is deliberate: the model's per-sample compute is trivial, so the goal isn't to parallelize a single sample's matmul (there's nothing worth splitting), it's to eliminate the ~6 kernel-launch overheads Phase 5 measured. Parallelism instead comes from the batch dimension, which is embarrassingly parallel — thread `i` handles sample `i` with zero cross-thread dependency.

**Grid/block configuration:** 128 threads per block, `ceil(batch_size / 128)` blocks. 128 is a standard warp-aligned block size (4 warps) that keeps occupancy reasonable without over-provisioning for a kernel this lightweight.

**Memory access pattern:** every thread reads the same weight matrices (`w1`, `w2`, `w3`) from global memory independently. This is intentional, not an oversight — the weights (64×29 + 32×64 + 32×1 floats ≈ 10.5KB total) are small enough to be served almost entirely from L2 cache across the whole thread grid, so redundant reads are cheap. Shared-memory caching of weights was considered and rejected: at this problem size, the added kernel complexity (cooperative loads, `__syncthreads()` barriers) would cost more in bookkeeping than the L2 cache misses it would save.

**Synchronization:** none needed inside the kernel — each thread's work is fully independent (no shared-memory writes between threads), so there's no intra-kernel `__syncthreads()`. The caller synchronizes via `torch.cuda.synchronize()` / `cudaDeviceSynchronize()` around the whole launch, same as every other backend in this project's benchmark harness.

**Numerical correctness:** weights are read directly in PyTorch's `nn.Linear` layout (`[out_features, in_features]`, row-major) with no transpose, so there's no layout-conversion bug surface. Accumulation is done in `float32` throughout, matching the trained model's precision.

## Correctness validation

Two independent checks, both passing:

1. **`cuda/kernel_correctness_test.cu`** — a standalone C++/CUDA executable (no PyTorch) that runs a plain-C++ reference forward pass (mirroring the kernel's math line-for-line) against the compiled kernel on the same exported weights and a fixed-seed random batch. Result: **max absolute difference 4.2e-7** (tolerance 1e-5) — PASSED.
2. **`tests/test_custom_cuda.py`** — 9 pytest tests comparing the kernel's output (via the PyTorch extension) against a CPU `FraudMLP` loaded with identical weights, parametrized across all 5 benchmark batch sizes. All pass with `atol=1e-5, rtol=1e-4`.

## Measured performance

Same batch sizes, warm-up (20 iters), and measurement count (200 iters) as the CPU and PyTorch-GPU benchmarks, run through the identical harness (`src/inference/benchmark.py`) for direct comparability.

| Batch | CPU (Phase 3) | PyTorch GPU (Phase 4) | **Custom CUDA kernel** | vs PyTorch GPU | vs CPU |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.070 ms | 0.239 ms | 0.091 ms | **2.6x faster** | 1.3x slower |
| 32 | 0.075 ms | 0.234 ms | 0.091 ms | **2.6x faster** | 1.2x slower |
| 128 | 0.090 ms | 0.237 ms | 0.101 ms | **2.3x faster** | 1.1x slower |
| 512 | 0.138 ms | 0.238 ms | 0.110 ms | **2.2x faster** | **1.3x faster** |
| 1024 | 0.161 ms | 0.274 ms | 0.118 ms | **2.3x faster** | **1.4x faster** |

Raw data: `benchmarks/results/custom_cuda_results.json`. Standalone (no-PyTorch) device-only kernel timing from `kernel_correctness_test.cu`: **0.030 ms/call at batch=1024**, vs Phase 5's measured `gpu_compute` stage of 0.231ms for the same batch size — consistent with launch-overhead elimination being the actual mechanism, not a different one.

## Honest conclusion

**The optimization worked against the target it was built for** (PyTorch's own GPU dispatch: 2.2x-2.6x faster at every batch size, confirming Phase 5's diagnosis was correct) **but did not uniformly beat the CPU baseline.** There's a real crossover: CPU still wins at batch sizes 1-128, the custom kernel wins at 512-1024. This is reported as measured, not rounded up to "GPU wins" or explained away — a 4,033-parameter model's CPU inference is fast enough that beating it requires the batch to be large enough to amortize the GPU's fixed per-call overhead (H2D/D2H transfer, kernel launch, Python/PyTorch dispatch into the extension), which the fused kernel reduced substantially but did not eliminate entirely.
