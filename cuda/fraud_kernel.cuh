// Fused forward-pass kernel for the FraudMLP architecture (29 -> 64 -> 32 -> 1).
//
// Specific to this model's fixed architecture rather than a generic MLP
// engine: PyTorch eager mode issues ~6 separate kernel launches per forward
// call for a network this small (see docs/bottleneck_analysis.md), so this
// kernel fuses all three Linear layers + 2 ReLUs + sigmoid into one launch.
// A generic multi-layer engine would reintroduce the same per-layer
// dispatch overhead internally.
#pragma once

// Dimensions match src/models/model.py's ModelConfig defaults exactly.
constexpr int FRAUD_INPUT_DIM = 29;
constexpr int FRAUD_HIDDEN1 = 64;
constexpr int FRAUD_HIDDEN2 = 32;

// Launches the fused forward-pass kernel on the given CUDA stream.
//
// All pointers are device pointers (already on GPU) -- host<->device
// transfer is the caller's responsibility (see src/inference/custom_cuda.py).
//
// Weight layout matches PyTorch's nn.Linear convention: W[out_features, in_features]
// row-major, i.e. W1 is [FRAUD_HIDDEN1, FRAUD_INPUT_DIM], etc.
void launch_fraud_mlp_forward(
    const float* x,       // [batch_size, FRAUD_INPUT_DIM]
    const float* w1,      // [FRAUD_HIDDEN1, FRAUD_INPUT_DIM]
    const float* b1,      // [FRAUD_HIDDEN1]
    const float* w2,      // [FRAUD_HIDDEN2, FRAUD_HIDDEN1]
    const float* b2,      // [FRAUD_HIDDEN2]
    const float* w3,      // [1, FRAUD_HIDDEN2]
    const float* b3,      // [1]
    float* out_probs,     // [batch_size]  (post-sigmoid probability)
    int batch_size,
    cudaStream_t stream
);
