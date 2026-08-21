// Standalone correctness + benchmark test for the fused FraudMLP CUDA kernel.
//
// Loads the exported trained weights (models/exported/*.bin), runs a plain
// C++ CPU reference forward pass and the CUDA kernel on the same random
// input batch, and reports:
//   1. Max absolute difference between CPU reference and CUDA kernel output
//      (correctness -- must be within float32 tolerance)
//   2. Wall-clock timing for both (pure C++/CUDA, no PyTorch involved)
//
// Usage: ./kernel_correctness_test <path-to-models/exported>

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <vector>

#include "fraud_kernel.cuh"

namespace {

std::vector<float> load_bin(const std::string& path, size_t expected_count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", path.c_str());
        exit(1);
    }
    std::vector<float> data(expected_count);
    f.read(reinterpret_cast<char*>(data.data()), expected_count * sizeof(float));
    if (!f) {
        fprintf(stderr, "Failed to read %zu floats from %s\n", expected_count, path.c_str());
        exit(1);
    }
    return data;
}

// Plain C++ reference forward pass -- deliberately mirrors the CUDA kernel's
// math line-for-line so any divergence is a real bug, not a different
// algorithm.
void cpu_reference_forward(
    const std::vector<float>& x, const std::vector<float>& w1, const std::vector<float>& b1,
    const std::vector<float>& w2, const std::vector<float>& b2,
    const std::vector<float>& w3, const std::vector<float>& b3,
    std::vector<float>& out, int batch_size
) {
    std::vector<float> h1(FRAUD_HIDDEN1), h2(FRAUD_HIDDEN2);
    for (int n = 0; n < batch_size; n++) {
        const float* x_row = &x[n * FRAUD_INPUT_DIM];
        for (int i = 0; i < FRAUD_HIDDEN1; i++) {
            float sum = b1[i];
            for (int j = 0; j < FRAUD_INPUT_DIM; j++) sum += w1[i * FRAUD_INPUT_DIM + j] * x_row[j];
            h1[i] = std::max(sum, 0.0f);
        }
        for (int i = 0; i < FRAUD_HIDDEN2; i++) {
            float sum = b2[i];
            for (int j = 0; j < FRAUD_HIDDEN1; j++) sum += w2[i * FRAUD_HIDDEN1 + j] * h1[j];
            h2[i] = std::max(sum, 0.0f);
        }
        float out_sum = b3[0];
        for (int j = 0; j < FRAUD_HIDDEN2; j++) out_sum += w3[j] * h2[j];
        out[n] = 1.0f / (1.0f + std::exp(-out_sum));
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string export_dir = argc > 1 ? argv[1] : "../../models/exported";
    const int batch_size = 1024;
    const int warmup_iters = 20;
    const int measure_iters = 200;

    auto w1 = load_bin(export_dir + "/w1.bin", FRAUD_HIDDEN1 * FRAUD_INPUT_DIM);
    auto b1 = load_bin(export_dir + "/b1.bin", FRAUD_HIDDEN1);
    auto w2 = load_bin(export_dir + "/w2.bin", FRAUD_HIDDEN2 * FRAUD_HIDDEN1);
    auto b2 = load_bin(export_dir + "/b2.bin", FRAUD_HIDDEN2);
    auto w3 = load_bin(export_dir + "/w3.bin", FRAUD_HIDDEN2);
    auto b3 = load_bin(export_dir + "/b3.bin", 1);

    // Deterministic random input batch (seed=42, matching the Python benchmarks)
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x(batch_size * FRAUD_INPUT_DIM);
    for (auto& v : x) v = dist(rng);

    // --- Correctness ---
    std::vector<float> cpu_out(batch_size);
    cpu_reference_forward(x, w1, b1, w2, b2, w3, b3, cpu_out, batch_size);

    float *d_x, *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3, *d_out;
    cudaMalloc(&d_x, x.size() * sizeof(float));
    cudaMalloc(&d_w1, w1.size() * sizeof(float));
    cudaMalloc(&d_b1, b1.size() * sizeof(float));
    cudaMalloc(&d_w2, w2.size() * sizeof(float));
    cudaMalloc(&d_b2, b2.size() * sizeof(float));
    cudaMalloc(&d_w3, w3.size() * sizeof(float));
    cudaMalloc(&d_b3, b3.size() * sizeof(float));
    cudaMalloc(&d_out, batch_size * sizeof(float));

    cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w1, w1.data(), w1.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, w2.data(), w2.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3, w3.data(), w3.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, b3.data(), b3.size() * sizeof(float), cudaMemcpyHostToDevice);

    launch_fraud_mlp_forward(d_x, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_out, batch_size, 0);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    std::vector<float> gpu_out(batch_size);
    cudaMemcpy(gpu_out.data(), d_out, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    float max_abs_diff = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        max_abs_diff = std::max(max_abs_diff, std::fabs(cpu_out[i] - gpu_out[i]));
    }
    printf("batch_size=%d, max_abs_diff (CPU ref vs CUDA kernel) = %.8f\n", batch_size, max_abs_diff);
    const float tolerance = 1e-5f;
    bool correct = max_abs_diff < tolerance;
    printf("%s (tolerance=%.0e)\n", correct ? "CORRECTNESS: PASSED" : "CORRECTNESS: FAILED", tolerance);

    // --- Benchmark ---
    for (int i = 0; i < warmup_iters; i++) {
        launch_fraud_mlp_forward(d_x, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_out, batch_size, 0);
    }
    cudaDeviceSynchronize();

    auto gpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < measure_iters; i++) {
        launch_fraud_mlp_forward(d_x, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_out, batch_size, 0);
    }
    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count() / measure_iters;

    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < measure_iters; i++) {
        cpu_reference_forward(x, w1, b1, w2, b2, w3, b3, cpu_out, batch_size);
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count() / measure_iters;

    printf("\nStandalone C++/CUDA timing (no PyTorch, batch_size=%d, %d iters):\n", batch_size, measure_iters);
    printf("  CPU reference (plain C++):   %.4f ms/call\n", cpu_ms);
    printf("  CUDA kernel (device only):   %.4f ms/call\n", gpu_ms);

    cudaFree(d_x); cudaFree(d_w1); cudaFree(d_b1); cudaFree(d_w2);
    cudaFree(d_b2); cudaFree(d_w3); cudaFree(d_b3); cudaFree(d_out);

    return correct ? 0 : 1;
}
