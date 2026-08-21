#include "inference.hpp"
#include "fraud_kernel.cuh"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace {

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error during " << what << ": " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

std::vector<float> read_binary_floats(const std::string& path, size_t expected_count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Failed to open weight file: " + path);
    }
    std::vector<float> data(expected_count);
    f.read(reinterpret_cast<char*>(data.data()), expected_count * sizeof(float));
    if (!f) {
        throw std::runtime_error("Failed to read expected " + std::to_string(expected_count) +
                                  " floats from: " + path);
    }
    return data;
}

float read_threshold(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Failed to open threshold file: " + path);
    }
    float value;
    f >> value;
    if (!f) {
        throw std::runtime_error("Failed to parse threshold from: " + path);
    }
    return value;
}

float* upload_to_device(const std::vector<float>& host_data, const char* what) {
    float* device_ptr = nullptr;
    check_cuda(cudaMalloc(&device_ptr, host_data.size() * sizeof(float)),
               (std::string("cudaMalloc for ") + what).c_str());
    check_cuda(cudaMemcpy(device_ptr, host_data.data(), host_data.size() * sizeof(float),
                           cudaMemcpyHostToDevice),
               (std::string("cudaMemcpy for ") + what).c_str());
    return device_ptr;
}

}  // namespace

FraudPredictor::FraudPredictor(const std::string& weights_dir, int initial_batch_capacity) {
    auto w1 = read_binary_floats(weights_dir + "/w1.bin", FRAUD_HIDDEN1 * FRAUD_INPUT_DIM);
    auto b1 = read_binary_floats(weights_dir + "/b1.bin", FRAUD_HIDDEN1);
    auto w2 = read_binary_floats(weights_dir + "/w2.bin", FRAUD_HIDDEN2 * FRAUD_HIDDEN1);
    auto b2 = read_binary_floats(weights_dir + "/b2.bin", FRAUD_HIDDEN2);
    auto w3 = read_binary_floats(weights_dir + "/w3.bin", FRAUD_HIDDEN2);
    auto b3 = read_binary_floats(weights_dir + "/b3.bin", 1);
    decision_threshold_ = read_threshold(weights_dir + "/threshold.txt");

    static_assert(FRAUD_PREDICTOR_INPUT_DIM == FRAUD_INPUT_DIM,
                  "cpp/inference.hpp and cuda/fraud_kernel.cuh input dims must match");

    d_w1_ = upload_to_device(w1, "w1");
    d_b1_ = upload_to_device(b1, "b1");
    d_w2_ = upload_to_device(w2, "w2");
    d_b2_ = upload_to_device(b2, "b2");
    d_w3_ = upload_to_device(w3, "w3");
    d_b3_ = upload_to_device(b3, "b3");

    ensure_capacity(initial_batch_capacity);
}

FraudPredictor::~FraudPredictor() {
    cudaFree(d_w1_);
    cudaFree(d_b1_);
    cudaFree(d_w2_);
    cudaFree(d_b2_);
    cudaFree(d_w3_);
    cudaFree(d_b3_);
    cudaFree(d_x_);
    cudaFree(d_out_);
}

void FraudPredictor::ensure_capacity(int batch_size) {
    if (batch_size <= batch_capacity_) {
        return;
    }
    if (d_x_) cudaFree(d_x_);
    if (d_out_) cudaFree(d_out_);

    check_cuda(cudaMalloc(&d_x_, static_cast<size_t>(batch_size) * FRAUD_PREDICTOR_INPUT_DIM * sizeof(float)),
               "resizing input buffer");
    check_cuda(cudaMalloc(&d_out_, static_cast<size_t>(batch_size) * sizeof(float)), "resizing output buffer");
    batch_capacity_ = batch_size;
}

std::vector<float> FraudPredictor::predict_batch_flat(const float* flat_features, int batch_size) {
    ensure_capacity(batch_size);

    check_cuda(cudaMemcpy(d_x_, flat_features,
                           static_cast<size_t>(batch_size) * FRAUD_PREDICTOR_INPUT_DIM * sizeof(float),
                           cudaMemcpyHostToDevice),
               "copying input batch to device");

    launch_fraud_mlp_forward(d_x_, d_w1_, d_b1_, d_w2_, d_b2_, d_w3_, d_b3_, d_out_, batch_size, 0);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "kernel execution");

    std::vector<float> probs(batch_size);
    check_cuda(cudaMemcpy(probs.data(), d_out_, static_cast<size_t>(batch_size) * sizeof(float),
                           cudaMemcpyDeviceToHost),
               "copying output batch to host");
    return probs;
}

PredictionResult FraudPredictor::predict(const std::vector<float>& features) {
    if (static_cast<int>(features.size()) != FRAUD_PREDICTOR_INPUT_DIM) {
        throw std::invalid_argument("Expected " + std::to_string(FRAUD_PREDICTOR_INPUT_DIM) +
                                     " features, got " + std::to_string(features.size()));
    }
    auto probs = predict_batch_flat(features.data(), 1);
    return {probs[0], probs[0] >= decision_threshold_};
}

std::vector<PredictionResult> FraudPredictor::predict_batch(const std::vector<std::vector<float>>& batch) {
    const int batch_size = static_cast<int>(batch.size());
    std::vector<float> flat(static_cast<size_t>(batch_size) * FRAUD_PREDICTOR_INPUT_DIM);
    for (int i = 0; i < batch_size; i++) {
        if (static_cast<int>(batch[i].size()) != FRAUD_PREDICTOR_INPUT_DIM) {
            throw std::invalid_argument("Row " + std::to_string(i) + ": expected " +
                                         std::to_string(FRAUD_PREDICTOR_INPUT_DIM) + " features, got " +
                                         std::to_string(batch[i].size()));
        }
        std::copy(batch[i].begin(), batch[i].end(), flat.begin() + i * FRAUD_PREDICTOR_INPUT_DIM);
    }

    auto probs = predict_batch_flat(flat.data(), batch_size);

    std::vector<PredictionResult> results(batch_size);
    for (int i = 0; i < batch_size; i++) {
        results[i] = {probs[i], probs[i] >= decision_threshold_};
    }
    return results;
}
