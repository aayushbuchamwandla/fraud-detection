// C++ inference demo + benchmark.
//
// Two modes:
//   ./fraud_cpp_demo <weights_dir> <samples_csv>
//       Loads real labeled transactions and prints predictions vs ground truth.
//   ./fraud_cpp_demo <weights_dir> --benchmark <output_json>
//       Runs the same batch-size sweep/methodology as the Python benchmarks
//       (benchmarks/benchmark_cpu.py etc.) and writes a matching JSON schema
//       so results are directly comparable in the combined benchmark table.

#include "inference.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct SampleTransaction {
    int test_split_index;
    int true_label;
    std::vector<float> features;
};

std::vector<SampleTransaction> load_samples(const std::string& csv_path) {
    std::ifstream f(csv_path);
    if (!f) {
        throw std::runtime_error("Failed to open samples CSV: " + csv_path);
    }

    std::vector<SampleTransaction> samples;
    std::string line;
    std::getline(f, line);  // header

    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string cell;
        SampleTransaction s;

        std::getline(ss, cell, ',');
        s.test_split_index = std::stoi(cell);
        std::getline(ss, cell, ',');
        s.true_label = std::stoi(cell);
        while (std::getline(ss, cell, ',')) {
            s.features.push_back(std::stof(cell));
        }
        samples.push_back(std::move(s));
    }
    return samples;
}

void run_demo(FraudPredictor& predictor, const std::string& samples_csv) {
    auto samples = load_samples(samples_csv);
    printf("Loaded %zu real labeled transactions from %s\n\n", samples.size(), samples_csv.c_str());
    printf("%-6s %-12s %-10s %-14s %-12s %-8s\n", "idx", "true_label", "pred_prob", "pred_label",
           "latency_ms", "correct");

    int n_correct = 0;
    for (auto& s : samples) {
        auto start = std::chrono::high_resolution_clock::now();
        PredictionResult result = predictor.predict(s.features);
        auto end = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();

        bool correct = (result.is_fraud ? 1 : 0) == s.true_label;
        n_correct += correct;

        printf("%-6d %-12s %-10.6f %-14s %-12.4f %-8s\n", s.test_split_index,
               s.true_label ? "FRAUD" : "legitimate", result.fraud_probability,
               result.is_fraud ? "FRAUD" : "legitimate", latency_ms, correct ? "yes" : "no");
    }
    printf("\n%d/%zu predictions matched the true label.\n", n_correct, samples.size());
}

struct BenchResult {
    int batch_size;
    double mean_ms, median_ms, p95_ms, p99_ms, throughput;
};

BenchResult benchmark_batch_size(FraudPredictor& predictor, int batch_size, int input_dim) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x(static_cast<size_t>(batch_size) * input_dim);
    for (auto& v : x) v = dist(rng);

    const int warmup_iters = 20;
    const int measure_iters = 200;

    for (int i = 0; i < warmup_iters; i++) {
        predictor.predict_batch_flat(x.data(), batch_size);
    }

    std::vector<double> latencies_ms;
    latencies_ms.reserve(measure_iters);
    for (int i = 0; i < measure_iters; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        predictor.predict_batch_flat(x.data(), batch_size);
        auto end = std::chrono::high_resolution_clock::now();
        latencies_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    std::sort(latencies_ms.begin(), latencies_ms.end());
    double mean = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) / measure_iters;
    double median = latencies_ms[measure_iters / 2];
    double p95 = latencies_ms[static_cast<size_t>(measure_iters * 0.95) - 1];
    double p99 = latencies_ms[static_cast<size_t>(measure_iters * 0.99) - 1];
    double throughput = batch_size / (mean / 1000.0);

    return {batch_size, mean, median, p95, p99, throughput};
}

void run_benchmark(FraudPredictor& predictor, const std::string& output_json) {
    std::vector<int> batch_sizes = {1, 32, 128, 512, 1024};
    std::vector<BenchResult> results;

    printf("Benchmarking C++ CUDA inference (persistent device buffers, no PyTorch)\n\n");
    for (int bs : batch_sizes) {
        auto r = benchmark_batch_size(predictor, bs, predictor.input_dim());
        results.push_back(r);
        printf("batch=%5d | mean %8.4f ms | median %8.4f ms | p95 %8.4f ms | p99 %8.4f ms | throughput %10.1f samples/s\n",
               r.batch_size, r.mean_ms, r.median_ms, r.p95_ms, r.p99_ms, r.throughput);
    }

    std::time_t now = std::time(nullptr);
    char timestamp[32];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));

    std::ofstream out(output_json);
    out << "[\n";
    for (size_t i = 0; i < results.size(); i++) {
        auto& r = results[i];
        out << "  {\n"
            << "    \"implementation\": \"cpp_cuda\",\n"
            << "    \"batch_size\": " << r.batch_size << ",\n"
            << "    \"warmup_iters\": 20,\n"
            << "    \"measure_iters\": 200,\n"
            << "    \"mean_latency_ms\": " << r.mean_ms << ",\n"
            << "    \"median_latency_ms\": " << r.median_ms << ",\n"
            << "    \"p95_latency_ms\": " << r.p95_ms << ",\n"
            << "    \"p99_latency_ms\": " << r.p99_ms << ",\n"
            << "    \"throughput_samples_per_sec\": " << r.throughput << ",\n"
            << "    \"device\": \"cuda (C++, persistent buffers)\",\n"
            << "    \"python_version\": null,\n"
            << "    \"torch_version\": null,\n"
            << "    \"timestamp\": \"" << timestamp << "\"\n"
            << "  }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    out << "]\n";
    printf("\nSaved results to %s\n", output_json.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage:\n  %s <weights_dir> <samples_csv>\n  %s <weights_dir> --benchmark <output_json>\n",
                argv[0], argv[0]);
        return 1;
    }

    try {
        std::string weights_dir = argv[1];
        FraudPredictor predictor(weights_dir);
        printf("Model loaded from %s, decision_threshold=%.4f\n\n", weights_dir.c_str(),
               predictor.decision_threshold());

        std::string mode = argv[2];
        if (mode == "--benchmark") {
            if (argc < 4) {
                fprintf(stderr, "Usage: %s <weights_dir> --benchmark <output_json>\n", argv[0]);
                return 1;
            }
            run_benchmark(predictor, argv[3]);
        } else {
            run_demo(predictor, mode);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
