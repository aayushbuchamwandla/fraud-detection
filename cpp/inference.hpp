// C++ inference component for the fraud detection model.
//
// This is a real inference engine, not a wrapper that shells out to Python:
// it owns persistent GPU device buffers for the model weights (uploaded
// once at construction, not per call) and reuses input/output device
// buffers across calls, growing them only when a larger batch is requested.
// That's a genuine difference from the PyTorch extension path in Phase 6,
// which re-wraps a torch::Tensor (with its own allocator bookkeeping) on
// every predict() call.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

constexpr int FRAUD_PREDICTOR_INPUT_DIM = 29;

struct PredictionResult {
    float fraud_probability;
    bool is_fraud;
};

// FraudPredictor is non-copyable (owns raw CUDA device pointers) and
// non-movable (no move constructor defined; the implicit ones are suppressed
// by the user-declared destructor/copy operations). One instance per model.
class FraudPredictor {
public:
    // Loads weights + decision threshold from `weights_dir` (expects
    // w1.bin, b1.bin, w2.bin, b2.bin, w3.bin, b3.bin, threshold.txt --
    // see scripts/export_weights.py) and uploads them to the GPU once.
    // Throws std::runtime_error if any file is missing/unreadable or a
    // CUDA call fails.
    explicit FraudPredictor(const std::string& weights_dir, int initial_batch_capacity = 1024);

    ~FraudPredictor();

    FraudPredictor(const FraudPredictor&) = delete;
    FraudPredictor& operator=(const FraudPredictor&) = delete;

    // Single transaction. Throws std::invalid_argument if features.size()
    // != FRAUD_PREDICTOR_INPUT_DIM.
    PredictionResult predict(const std::vector<float>& features);

    // Multiple transactions in one kernel launch. Throws std::invalid_argument
    // if any row has the wrong feature count.
    std::vector<PredictionResult> predict_batch(const std::vector<std::vector<float>>& batch);

    // Lower-level entry point used by predict()/predict_batch() and by the
    // benchmark harness (main.cpp): takes a flat, row-major
    // [batch_size x FRAUD_PREDICTOR_INPUT_DIM] array and returns raw
    // probabilities (no threshold applied). Grows the internal device
    // buffers if batch_size exceeds current capacity.
    std::vector<float> predict_batch_flat(const float* flat_features, int batch_size);

    float decision_threshold() const { return decision_threshold_; }
    int input_dim() const { return FRAUD_PREDICTOR_INPUT_DIM; }

private:
    void ensure_capacity(int batch_size);

    float decision_threshold_;

    // Persistent device weight buffers (allocated once, freed in destructor)
    float* d_w1_ = nullptr;
    float* d_b1_ = nullptr;
    float* d_w2_ = nullptr;
    float* d_b2_ = nullptr;
    float* d_w3_ = nullptr;
    float* d_b3_ = nullptr;

    // Reusable input/output device buffers, grown on demand
    float* d_x_ = nullptr;
    float* d_out_ = nullptr;
    int batch_capacity_ = 0;
};
