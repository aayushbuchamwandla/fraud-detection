"""
Generates a combined markdown benchmark table from the actual JSON results
-- this is also a cross-check on every hand-copied number in the README:
regenerating this table and diffing it against what's in the README is how
you'd catch a transcription error, not just trust that copy-paste went fine.

Usage:
    python benchmarks/generate_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

IMPLEMENTATIONS = [
    ("cpu_pytorch", "CPU (PyTorch)", "cpu_results.json"),
    ("pytorch_cuda", "PyTorch GPU", "gpu_results.json"),
    ("custom_cuda_kernel", "Custom CUDA kernel", "custom_cuda_results.json"),
    ("cpp_cuda", "C++ (persistent buffers)", "cpp_results.json"),
    ("tensorrt", "TensorRT", "tensorrt_results.json"),
]

BATCH_SIZES = [1, 32, 128, 512, 1024]


def main() -> None:
    all_results = {}
    for key, label, filename in IMPLEMENTATIONS:
        path = RESULTS_DIR / filename
        if path.exists():
            with open(path) as f:
                all_results[key] = {r["batch_size"]: r for r in json.load(f)}

    lines = []
    lines.append("# Benchmark summary (auto-generated from benchmarks/results/*.json -- do not hand-edit)")
    lines.append("")
    lines.append("Mean latency (ms) by batch size:")
    lines.append("")
    header = "| Implementation | " + " | ".join(f"batch={b}" for b in BATCH_SIZES) + " |"
    sep = "|---|" + "---:|" * len(BATCH_SIZES)
    lines.append(header)
    lines.append(sep)
    for key, label, _ in IMPLEMENTATIONS:
        if key not in all_results:
            continue
        row = [label]
        for b in BATCH_SIZES:
            r = all_results[key].get(b)
            row.append(f"{r['mean_latency_ms']:.3f} ms" if r else "-")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("Throughput (samples/sec) by batch size:")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for key, label, _ in IMPLEMENTATIONS:
        if key not in all_results:
            continue
        row = [label]
        for b in BATCH_SIZES:
            r = all_results[key].get(b)
            row.append(f"{r['throughput_samples_per_sec']:,.0f}" if r else "-")
        lines.append("| " + " | ".join(row) + " |")

    out_path = RESULTS_DIR / "summary_table.md"
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
