// Fused forward-pass kernel for FraudMLP (29 -> 64 -> 32 -> 1).
//
// One CUDA thread per transaction: each thread independently computes the
// entire 3-layer forward pass for its sample (~3,936 multiply-adds total --
// trivial per-thread work), so parallelism comes from the batch dimension,
// not from splitting a single sample's matmul across threads. This is the
// right granularity here because the whole point (see
// docs/bottleneck_analysis.md) is eliminating per-layer KERNEL LAUNCH
// overhead, not speeding up an individual matmul that's already tiny.
//
// Weight matrices are small enough (64x29, 32x64, 32x1 floats) that every
// thread re-reads them from global memory; L2 cache handles this well since
// all threads in a launch read the exact same weight values.

#include <cuda_runtime.h>
#include <math.h>
#include "fraud_kernel.cuh"

__global__ void fraud_mlp_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w1,
    const float* __restrict__ b1,
    const float* __restrict__ w2,
    const float* __restrict__ b2,
    const float* __restrict__ w3,
    const float* __restrict__ b3,
    float* __restrict__ out_probs,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* x_row = x + idx * FRAUD_INPUT_DIM;

    // Layer 1: Linear(29 -> 64) + ReLU
    float h1[FRAUD_HIDDEN1];
    #pragma unroll
    for (int i = 0; i < FRAUD_HIDDEN1; i++) {
        float sum = b1[i];
        const float* w1_row = w1 + i * FRAUD_INPUT_DIM;
        #pragma unroll
        for (int j = 0; j < FRAUD_INPUT_DIM; j++) {
            sum += w1_row[j] * x_row[j];
        }
        h1[i] = fmaxf(sum, 0.0f);
    }

    // Layer 2: Linear(64 -> 32) + ReLU
    float h2[FRAUD_HIDDEN2];
    #pragma unroll
    for (int i = 0; i < FRAUD_HIDDEN2; i++) {
        float sum = b2[i];
        const float* w2_row = w2 + i * FRAUD_HIDDEN1;
        #pragma unroll
        for (int j = 0; j < FRAUD_HIDDEN1; j++) {
            sum += w2_row[j] * h1[j];
        }
        h2[i] = fmaxf(sum, 0.0f);
    }

    // Layer 3: Linear(32 -> 1) + sigmoid
    float out_sum = b3[0];
    #pragma unroll
    for (int j = 0; j < FRAUD_HIDDEN2; j++) {
        out_sum += w3[j] * h2[j];
    }
    out_probs[idx] = 1.0f / (1.0f + expf(-out_sum));
}

void launch_fraud_mlp_forward(
    const float* x,
    const float* w1,
    const float* b1,
    const float* w2,
    const float* b2,
    const float* w3,
    const float* b3,
    float* out_probs,
    int batch_size,
    cudaStream_t stream
) {
    const int threads_per_block = 128;
    const int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    fraud_mlp_forward_kernel<<<blocks, threads_per_block, 0, stream>>>(
        x, w1, b1, w2, b2, w3, b3, out_probs, batch_size
    );
}
