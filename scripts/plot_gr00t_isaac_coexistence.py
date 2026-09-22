#!/usr/bin/env python3
"""Plot the measured GR00T + SONIC + Isaac GPU coexistence probe."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/benchmark/performance"


def plot_probe(result: dict, telemetry: list[dict], output: Path) -> None:
    attempt = result.get("gpu_attempt", {})
    if (
        result.get("status") != "passed"
        or result.get("policy_device") != "cuda:0"
        or int(attempt.get("health", {}).get("infer_count", 0)) < 1
        or int(result.get("measured_peak_compute_vram_mib", 0)) > int(result.get("maximum_peak_compute_vram_mib", 0))
    ):
        raise ValueError("input is not a passed real-inference CUDA coexistence probe")
    samples = [row for row in telemetry if "compute_memory_used_mib" in row and "gpu_utilization_percent" in row]
    if not samples:
        raise ValueError("telemetry has no valid GPU samples")
    t0 = datetime.fromisoformat(samples[0]["recorded_at_utc"])
    seconds = [(datetime.fromisoformat(row["recorded_at_utc"]) - t0).total_seconds() for row in samples]
    compute = [float(row["compute_memory_used_mib"]) for row in samples]
    total = [float(row["gpu_memory_used_mib"]) for row in samples]
    util = [float(row["gpu_utilization_percent"]) for row in samples]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(seconds, compute, color="#1c7194", linewidth=2, label="Compute VRAM")
    ax.plot(seconds, total, color="#64748b", linewidth=1.3, alpha=0.75, label="Total GPU VRAM")
    ax.axhline(float(result["maximum_peak_compute_vram_mib"]), color="#bc643d", linestyle="--", linewidth=1.3,
               label="Locked 15,500 MiB limit")
    ax.set_ylabel("GPU memory (MiB)")
    ax.set_xlabel("Seconds from probe start")
    ax.set_ylim(0, max(max(total), 15500) * 1.08)
    ax.grid(alpha=0.23)
    ax.set_axisbelow(True)
    other = ax.twinx()
    other.plot(seconds, util, color="#518b67", alpha=0.5, linewidth=1.2, label="GPU utilization")
    other.set_ylabel("GPU utilization (%)")
    other.set_ylim(0, 105)
    lines, labels = ax.get_legend_handles_labels()
    more_lines, more_labels = other.get_legend_handles_labels()
    ax.legend(lines + more_lines, labels + more_labels, loc="upper left", fontsize=9)
    ax.set_title(
        f"GR00T + camera + SONIC + Isaac: {attempt['health']['infer_count']} real inferences, "
        f"{result['measured_peak_compute_vram_mib']:,} MiB compute peak"
    )
    fig.text(0.5, 0.025, "40-step hardware coexistence probe; timeout is not a task-success benchmark.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=0.1, right=0.88, top=0.88, bottom=0.16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, metadata={"Software": "HRVLA coexistence plotter"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=BASE / "gr00t_isaac_coexistence_v2.json")
    parser.add_argument("--telemetry", type=Path, default=BASE / "gr00t_isaac_coexistence_v2_telemetry.jsonl")
    parser.add_argument("--output", type=Path, default=BASE / "gr00t_isaac_coexistence_v2.png")
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    telemetry = [json.loads(line) for line in args.telemetry.read_text(encoding="utf-8").splitlines() if line.strip()]
    plot_probe(result, telemetry, args.output)
    print(f"[coexistence-plot] output={args.output}")


if __name__ == "__main__":
    main()
