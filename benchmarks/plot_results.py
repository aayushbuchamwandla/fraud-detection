"""
Phase 12: generate benchmark comparison plots from the ACTUAL JSON results
already collected in benchmarks/results/ -- every number plotted here comes
from a real benchmark run (Phases 3/4/6/7/8), nothing is typed in by hand.

Color: each implementation gets one fixed hue, in the same order across
every chart in this script (color follows the entity, not its rank) --
categorical palette validated colorblind-safe (see the dataviz skill's
references/palette.md).

Usage:
    python benchmarks/plot_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# implementation key -> (display label, result file)
IMPLEMENTATIONS = [
    ("cpu_pytorch", "CPU (PyTorch)", "cpu_results.json"),
    ("pytorch_cuda", "PyTorch GPU", "gpu_results.json"),
    ("custom_cuda_kernel", "Custom CUDA kernel", "custom_cuda_results.json"),
    ("cpp_cuda", "C++ (persistent buffers)", "cpp_results.json"),
    ("tensorrt", "TensorRT", "tensorrt_results.json"),
]

# Fixed categorical palette (blue, orange, aqua, yellow, magenta) -- CVD-safe
# ordering, assigned once and reused identically across every chart so the
# same implementation is always the same color.
COLORS = {
    "cpu_pytorch": "#2a78d6",
    "pytorch_cuda": "#eb6834",
    "custom_cuda_kernel": "#1baf7a",
    "cpp_cuda": "#eda100",
    "tensorrt": "#e87ba4",
}

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def load_results() -> dict[str, list[dict]]:
    """Returns {impl_key: [result_dict, ...]}, skipping any missing files
    gracefully (e.g. TensorRT results on a machine that hasn't built one)."""
    data = {}
    for key, _label, filename in IMPLEMENTATIONS:
        path = RESULTS_DIR / filename
        if path.exists():
            with open(path) as f:
                data[key] = json.load(f)
        else:
            print(f"(skipping {key}: {path} not found)")
    return data


def label_of(key: str) -> str:
    return next(label for k, label, _ in IMPLEMENTATIONS if k == key)


def plot_latency_by_implementation(data: dict, batch_size: int = 1024) -> None:
    """Bar chart: mean latency per implementation, at one representative batch size."""
    keys = [k for k in data if any(r["batch_size"] == batch_size for r in data[k])]
    values = [next(r["mean_latency_ms"] for r in data[k] if r["batch_size"] == batch_size) for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(keys)), values, color=[COLORS[k] for k in keys], width=0.6)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([label_of(k) for k in keys], rotation=20, ha="right")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title(f"Inference latency by implementation (batch size = {batch_size})", color=INK, fontsize=13)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02, f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, color=INK,
        )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "latency_by_implementation.png", dpi=150)
    plt.close(fig)


def plot_throughput_by_implementation(data: dict, batch_size: int = 1024) -> None:
    """Bar chart: throughput per implementation, at the same representative batch size."""
    keys = [k for k in data if any(r["batch_size"] == batch_size for r in data[k])]
    values = [
        next(r["throughput_samples_per_sec"] for r in data[k] if r["batch_size"] == batch_size) for k in keys
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(keys)), values, color=[COLORS[k] for k in keys], width=0.6)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([label_of(k) for k in keys], rotation=20, ha="right")
    ax.set_ylabel("Throughput (samples/sec)")
    ax.set_title(f"Inference throughput by implementation (batch size = {batch_size})", color=INK, fontsize=13)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02, f"{val:,.0f}",
            ha="center", va="bottom", fontsize=9, color=INK,
        )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "throughput_by_implementation.png", dpi=150)
    plt.close(fig)


def plot_latency_across_batch_sizes(data: dict) -> None:
    """Line chart: mean latency vs batch size, one line per implementation."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for key in data:
        results = sorted(data[key], key=lambda r: r["batch_size"])
        batch_sizes = [r["batch_size"] for r in results]
        latencies = [r["mean_latency_ms"] for r in results]
        ax.plot(
            batch_sizes, latencies, marker="o", markersize=6, linewidth=2,
            color=COLORS[key], label=label_of(key),
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 32, 128, 512, 1024])
    ax.set_xticklabels(["1", "32", "128", "512", "1024"])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Latency across batch sizes, by implementation", color=INK, fontsize=13)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "latency_across_batch_sizes.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results()

    if not data:
        print("No benchmark results found. Run the benchmarks/benchmark_*.py scripts first.")
        return

    plot_latency_by_implementation(data)
    plot_throughput_by_implementation(data)
    plot_latency_across_batch_sizes(data)

    print(f"Saved 3 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
