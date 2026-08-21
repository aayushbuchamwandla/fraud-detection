# Phase 5: Pipeline Bottleneck Analysis

Phase 4 found that PyTorch CUDA inference is 1.7x-3.4x *slower* than CPU at every tested batch size for this model. Before writing a custom CUDA kernel to "fix" that, this phase measures **where the GPU pipeline actually spends its time**, so any optimization targets a real, measured bottleneck rather than a guess.

## Method

`benchmarks/profile_pipeline.py` times five stages of a single inference call separately, with `torch.cuda.synchronize()` inserted between stages so each measurement reflects only that stage's actual wall-clock cost (without synchronization, GPU work queued asynchronously would leak into the next stage's timer):

```
host tensor prep -> H2D transfer -> GPU compute (forward pass) -> D2H transfer -> postprocess (threshold)
```

20 warm-up iterations (CUDA context init, allocator caching) precede 200 measured iterations, at the same batch sizes used in Phase 3/4 (1, 32, 128, 512, 1024).

## Result

| Batch | host_prep | h2d_transfer | **gpu_compute** | d2h_transfer | postprocess | total |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.6% | 9.8% | **71.4%** | 8.5% | 5.7% | 0.360 ms |
| 32 | 2.0% | 10.7% | **73.1%** | 9.6% | 4.5% | 0.282 ms |
| 128 | 2.4% | 11.3% | **72.2%** | 9.5% | 4.5% | 0.285 ms |
| 512 | 2.8% | 12.3% | **71.0%** | 9.5% | 4.5% | 0.301 ms |
| 1024 | 3.5% | 13.5% | **69.9%** | 8.8% | 4.3% | 0.330 ms |

Raw data: `benchmarks/results/pipeline_profile.json`.

## Diagnosis

Two things stand out:

1. **`gpu_compute` dominates (~70-73% of total time) at every batch size.**
2. **`gpu_compute`'s absolute time is nearly flat** — 0.206ms at batch=32 vs 0.231ms at batch=1024, a 5x increase in batch size producing an 11% increase in compute time.

If the GPU were actually compute-bound on this workload, `gpu_compute` would scale with batch size (more rows -> more FLOPs -> more time). It doesn't. That's the signature of **fixed per-call launch overhead**, not arithmetic throughput.

The reason: `FraudMLP.forward()` is `Linear -> ReLU -> Linear -> ReLU -> Linear`, then a separate `sigmoid` call. In PyTorch eager mode, each of those ops is dispatched as its own CUDA kernel launch — roughly 6 kernel launches per forward call (3x `Linear`/GEMM, 2x `ReLU`, 1x `sigmoid`), each incurring CPU-side dispatch latency (Python/ATen dispatch, CUDA driver queuing) on the order of microseconds, independent of how much data the kernel actually processes. For matrices this small (29x64, 64x32, 32x1), the actual GEMM computation is a rounding error next to that fixed dispatch cost — so six small launches cost roughly six times the fixed overhead, regardless of batch size.

The H2D/D2H transfer stages show the same flat-with-batch-size pattern (10-14% and 9-10%) for the same underlying reason: at these data sizes (29 floats x batch_size), transfer time is dominated by the fixed per-`cudaMemcpy` launch cost, not by PCIe bandwidth (nowhere near saturated by a few KB).

## Conclusion: what's actually worth optimizing

**Kernel launch overhead in `gpu_compute`, not the arithmetic itself, is the real bottleneck** — and it's the one piece of this pipeline a custom CUDA kernel can directly address: fusing the three-layer MLP forward pass (`linear -> relu -> linear -> relu -> linear -> sigmoid`) into a single custom kernel replaces ~6 kernel launches with 1, eliminating roughly 5/6 of the dispatch overhead that Phase 5 measured as ~70% of total pipeline time.

This is stated as the target for Phase 6, not as a result — whether a fused kernel actually beats PyTorch's dispatch here (and whether the resulting GPU path finally beats the CPU baseline) is a benchmark question the next phase answers, not an assumption this one makes.
