#!/usr/bin/env python3
"""Render consolidated comparison tables and charts from retained experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = ["plan_once", "recursive", "best_of_n", "ttc", "adaptive_ttc"]
COLORS = {
    "plan_once": "#718096",
    "recursive": "#805ad5",
    "best_of_n": "#d69e2e",
    "ttc": "#c53030",
    "adaptive_ttc": "#2b6cb0",
}


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required experiment output is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/subtask"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.results_root.resolve()
    output = (args.output_dir or root / "comparison").resolve()
    output.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to render result charts") from exc

    cpu_sources = {
        0.15: root / "cpu-final-noise-015" / "summary.json",
        0.28: root / "cpu-final-primary-028" / "summary.json",
        0.45: root / "cpu-final-noise-045" / "summary.json",
    }
    cpu_rows = []
    for noise, path in cpu_sources.items():
        for method, values in load_json(path)["methods"].items():
            cpu_rows.append({"proposal_error_rate": noise, "method": method, **values})

    cpu_csv = output / "cpu_stress_matrix.csv"
    keys = [
        "proposal_error_rate",
        "method",
        "episodes",
        "task_success_rate",
        "mean_progress",
        "next_subtask_accuracy",
        "valid_selection_rate",
        "injected_failure_recovery_rate",
        "safety_fallback_rate",
        "ttc_route_rate",
        "mean_model_calls_per_decision",
        "mean_search_nodes_per_decision",
        "mean_planner_latency_ms",
    ]
    with cpu_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cpu_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for method in METHODS:
        rows = sorted(
            (row for row in cpu_rows if row["method"] == method),
            key=lambda row: row["proposal_error_rate"],
        )
        x = [row["proposal_error_rate"] for row in rows]
        axes[0].plot(
            x,
            [row["next_subtask_accuracy"] for row in rows],
            marker="o",
            label=method,
            color=COLORS[method],
        )
        axes[1].plot(
            x,
            [row["mean_model_calls_per_decision"] for row in rows],
            marker="o",
            label=method,
            color=COLORS[method],
        )
    axes[0].set(title="Robustness to proposal error", xlabel="Proposal error rate", ylabel="Accuracy")
    axes[0].set_ylim(0.88, 1.005)
    axes[1].set(title="Compute scaling", xlabel="Proposal error rate", ylabel="Model calls / decision")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)
    fig.savefig(output / "cpu_robustness.png", dpi=180)
    plt.close(fig)

    gpu = load_json(root / "cosmos-reason2-release" / "cosmos_summary.json")
    gpu_values = gpu["methods"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    labels = [method.replace("_", "\n") for method in METHODS]
    plots = [
        ("next_subtask_accuracy", "Canonical accuracy", (0, 1.05)),
        ("safety_fallback_rate", "Safety fallback rate", (0, 0.25)),
        ("mean_latency_ms", "Mean latency (ms)", None),
    ]
    for axis, (key, title, limits) in zip(axes, plots):
        values = [gpu_values[method][key] for method in METHODS]
        bars = axis.bar(labels, values, color=[COLORS[method] for method in METHODS])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        if limits:
            axis.set_ylim(*limits)
        axis.bar_label(bars, fmt="%.3f" if limits else "%.0f", fontsize=7)
    fig.savefig(output / "gpu_method_comparison.png", dpi=180)
    plt.close(fig)

    threshold_sources = {
        0.50: (
            root / "cpu-threshold-050" / "summary.json",
            root / "cosmos-threshold-050" / "cosmos_summary.json",
        ),
        0.72: (
            root / "cpu-final-primary-028" / "summary.json",
            root / "cosmos-reason2-release" / "cosmos_summary.json",
        ),
        0.85: (
            root / "cpu-threshold-085" / "summary.json",
            root / "cosmos-threshold-085" / "cosmos_summary.json",
        ),
    }
    threshold_rows = []
    for threshold, (cpu_path, gpu_path) in threshold_sources.items():
        for domain, path in (("symbolic_cpu", cpu_path), ("cosmos_gpu", gpu_path)):
            values = load_json(path)["methods"]["adaptive_ttc"]
            threshold_rows.append(
                {
                    "domain": domain,
                    "confidence_threshold": threshold,
                    "next_subtask_accuracy": values["next_subtask_accuracy"],
                    "mean_model_calls_per_decision": values["mean_model_calls_per_decision"],
                    "ttc_route_rate": values["ttc_route_rate"],
                    "safety_fallback_rate": values["safety_fallback_rate"],
                }
            )
    threshold_csv = output / "threshold_ablation.csv"
    with threshold_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(threshold_rows[0]))
        writer.writeheader()
        writer.writerows(threshold_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for domain, color in (("symbolic_cpu", "#2b6cb0"), ("cosmos_gpu", "#d69e2e")):
        rows = [row for row in threshold_rows if row["domain"] == domain]
        x = [row["confidence_threshold"] for row in rows]
        axes[0].plot(
            x,
            [row["next_subtask_accuracy"] for row in rows],
            marker="o",
            label=domain,
            color=color,
        )
        axes[1].plot(
            x,
            [row["mean_model_calls_per_decision"] for row in rows],
            marker="o",
            label=domain,
            color=color,
        )
    axes[0].set(title="Threshold quality", xlabel="Confidence threshold", ylabel="Accuracy")
    axes[0].set_ylim(0.86, 1.01)
    axes[1].set(title="Threshold compute", xlabel="Confidence threshold", ylabel="Model calls / decision")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(output / "threshold_ablation.png", dpi=180)
    plt.close(fig)

    consolidated = {
        "schema_version": 1,
        "cpu_sources": {str(key): str(value) for key, value in cpu_sources.items()},
        "cpu_rows": cpu_rows,
        "gpu_source": str(root / "cosmos-reason2-release" / "cosmos_summary.json"),
        "gpu": gpu,
        "threshold_rows": threshold_rows,
        "sonic_source": str(root / "sonic-regression-v3" / "metrics_eval.json"),
        "sonic": load_json(root / "sonic-regression-v3" / "metrics_eval.json"),
    }
    (output / "consolidated.json").write_text(
        json.dumps(consolidated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote consolidated results to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
