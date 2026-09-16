#!/usr/bin/env python3
"""Compile the frozen serial-vs-batched GR00T evaluator throughput evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


def _load(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


def _row_map(report: dict) -> dict[tuple[int, str], dict]:
    return {
        (int(row["trajectory_id"]), str(row["condition"])): row
        for row in report["rows"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/benchmark/readiness"),
    )
    args = parser.parse_args()
    candidates = [
        ("serial-32traj", "serial"),
        ("batch-2-workers-4", "batch 2 / workers 4"),
        ("batch-4-workers-4", "batch 4 / workers 4"),
        ("batch-8-workers-4", "batch 8 / workers 4"),
        ("batch-15-workers-8-16traj", "batch 15 / workers 8"),
        ("batch-30-workers-8-16traj", "batch 30 / workers 8"),
        ("batch-60-workers-8-16traj", "batch 60 / workers 8"),
        (
            "batch-30-workers-16-pending-1-telemetry-32traj",
            "batch 30 / workers 16",
        ),
    ]
    reports = {name: _load(args.profile_root / name) for name, _ in candidates}
    reference = reports["serial-32traj"]
    reference_rows = _row_map(reference)
    reference_seconds_per_trajectory = (
        float(reference["execution"]["wall_seconds"])
        / len(reference["trajectory_ids"])
    )
    rows = []
    for name, label in candidates:
        report = reports[name]
        candidate_rows = _row_map(report)
        shared = sorted(set(reference_rows) & set(candidate_rows))
        mse_deltas = [
            abs(float(reference_rows[key]["mse"]) - float(candidate_rows[key]["mse"]))
            for key in shared
        ]
        rate_deltas = [
            abs(
                float(reference_rows[key]["within_0_1_rate"])
                - float(candidate_rows[key]["within_0_1_rate"])
            )
            for key in shared
        ]
        seconds_per_trajectory = (
            float(report["execution"]["wall_seconds"])
            / len(report["trajectory_ids"])
        )
        rows.append(
            {
                "name": name,
                "label": label,
                "profile": report["execution"]["profile"],
                "trajectories": len(report["trajectory_ids"]),
                "seconds_per_trajectory": seconds_per_trajectory,
                "speedup_over_serial": reference_seconds_per_trajectory
                / seconds_per_trajectory,
                "peak_vram_gib": float(report["hardware"]["peak_vram_gib"]),
                "maximum_mse_delta_on_shared_rows": max(mse_deltas, default=0.0),
                "maximum_rate_delta_on_shared_rows": max(rate_deltas, default=0.0),
                "mean_mse_delta_on_shared_rows": statistics.fmean(mse_deltas)
                if mse_deltas
                else 0.0,
            }
        )
    selected = next(
        row
        for row in rows
        if row["name"] == "batch-30-workers-16-pending-1-telemetry-32traj"
    )
    telemetry_path = args.profile_root / selected["name"] / "telemetry.csv"
    telemetry = list(csv.DictReader(telemetry_path.open(encoding="utf-8")))

    def telemetry_values(key: str) -> list[float]:
        values = []
        for row in telemetry:
            try:
                values.append(float(row[key]))
            except (KeyError, TypeError, ValueError):
                pass
        return values

    gpu_utilization = telemetry_values("utilization_gpu")
    active_gpu_utilization = [value for value in gpu_utilization if value > 0]
    process_cpu = telemetry_values("process_cpu_percent")
    process_rss = telemetry_values("process_rss_kib")
    available_memory = telemetry_values("available_memory_kib")
    selected["telemetry"] = {
        "samples": len(telemetry),
        "maximum_gpu_utilization_percent": max(gpu_utilization),
        "mean_active_gpu_utilization_percent": statistics.fmean(active_gpu_utilization),
        "maximum_process_cpu_percent": max(process_cpu),
        "maximum_process_rss_gib": max(process_rss) / 2**20,
        "minimum_available_memory_gib": min(available_memory) / 2**20,
    }
    gate = {
        "maximum_mse_delta": 0.0001,
        "maximum_rate_delta": 0.002,
        "passes": (
            selected["maximum_mse_delta_on_shared_rows"] <= 0.0001
            and selected["maximum_rate_delta_on_shared_rows"] <= 0.002
        ),
    }
    summary = {
        "schema_version": 1,
        "claim_boundary": (
            "validation-only infrastructure profile; batched amortized timing is not "
            "a single-request policy-latency claim"
        ),
        "selected": selected["name"],
        "equivalence_gate": gate,
        "rows": rows,
    }
    if not gate["passes"]:
        raise RuntimeError("selected evaluator profile failed the equivalence gate")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "vla_evaluator_throughput.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["label"].replace(" / ", "\n") for row in rows]
    colors = ["#2f855a" if row is selected else "#718096" for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    bars = axes[0].bar(labels, [row["speedup_over_serial"] for row in rows], color=colors)
    axes[0].bar_label(bars, fmt="%.2fx")
    axes[0].set_ylabel("Speedup over serial")
    axes[0].set_title("Validation evaluator throughput")
    axes[0].grid(axis="y", alpha=0.25)
    bars = axes[1].bar(labels, [row["peak_vram_gib"] for row in rows], color=colors)
    axes[1].bar_label(bars, fmt="%.2f")
    axes[1].axhline(16.0, color="#c53030", linestyle="--", label="16 GiB budget")
    axes[1].set_ylabel("Peak allocated VRAM (GiB)")
    axes[1].set_title("Memory safety")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    figure.suptitle("Deterministic CPU-prefetch / GPU-batch evaluator profile")
    figure.savefig(args.output_dir / "vla_evaluator_throughput.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"selected": summary["selected"], "gate": gate}, sort_keys=True))


if __name__ == "__main__":
    main()
