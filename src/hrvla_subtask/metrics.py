"""Aggregate high-level planning metrics and produce reviewable chart data."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
from typing import Any, Iterable


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["method"])].append(record)
    output: dict[str, Any] = {"schema_version": 1, "methods": {}}
    for method, rows in sorted(grouped.items()):
        decisions = sum(int(row["decisions"]) for row in rows)
        injections = sum(int(row["injected_failures"]) for row in rows)
        calls = sum(int(row["model_calls"]) for row in rows)
        latencies = [
            float(row["planner_latency_ms"]) / max(1, int(row["decisions"])) for row in rows
        ]
        sorted_latency = sorted(latencies)
        p95_index = max(0, min(len(sorted_latency) - 1, round(0.95 * len(sorted_latency)) - 1))
        output["methods"][method] = {
            "episodes": len(rows),
            "task_success_rate": statistics.fmean(bool(row["success"]) for row in rows),
            "mean_progress": statistics.fmean(float(row["progress"]) for row in rows),
            "next_subtask_accuracy": _ratio(
                sum(int(row["correct_decisions"]) for row in rows), decisions
            ),
            "valid_selection_rate": _ratio(
                sum(int(row["valid_selected"]) for row in rows), decisions
            ),
            "injected_failure_recovery_rate": _ratio(
                sum(int(row["recovered_failures"]) for row in rows), injections
            ),
            "safety_fallback_rate": _ratio(
                sum(int(row["safety_fallbacks"]) for row in rows), decisions
            ),
            "hallucinated_candidates_per_decision": _ratio(
                sum(int(row["hallucinated_candidates"]) for row in rows), decisions
            ),
            "repeated_action_rate": _ratio(
                sum(int(row["repeated_actions"]) for row in rows), decisions
            ),
            "mean_steps": statistics.fmean(int(row["steps"]) for row in rows),
            "ttc_route_rate": _ratio(sum(int(row["ttc_decisions"]) for row in rows), decisions),
            "mean_search_nodes_per_decision": _ratio(
                sum(int(row["search_nodes"]) for row in rows), decisions
            ),
            "mean_model_calls_per_decision": _ratio(calls, decisions),
            "generated_tokens": sum(int(row["generated_tokens"]) for row in rows),
            "mean_planner_latency_ms": statistics.fmean(latencies),
            "p95_episode_mean_latency_ms": sorted_latency[p95_index],
        }
    return output


def write_chart_data(summary: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = summary["methods"]
    metric_names = sorted({key for values in rows.values() for key in values if key != "episodes"})
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "episodes", *metric_names])
        for method, values in rows.items():
            writer.writerow([method, values["episodes"], *(values.get(key) for key in metric_names)])


def plot_summary(summary: dict[str, Any], output_dir: str | Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional plotting environment
        raise RuntimeError("matplotlib is required to render benchmark charts") from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    methods = list(summary["methods"])
    labels = [item.replace("_", "\n") for item in methods]
    quality_metrics = [
        ("task_success_rate", "Task success"),
        ("mean_progress", "Progress"),
        ("next_subtask_accuracy", "Next-subtask accuracy"),
        ("injected_failure_recovery_rate", "Failure recovery"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, quality_metrics):
        values = [summary["methods"][method].get(key) or 0.0 for method in methods]
        bars = axis.bar(labels, values, color="#2b6cb0")
        axis.set_ylim(0, 1.05)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.3f", fontsize=8)
    quality_path = output / "planner_quality.png"
    fig.savefig(quality_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    compute_metrics = [
        ("mean_model_calls_per_decision", "Model calls / decision"),
        ("mean_search_nodes_per_decision", "Search nodes / decision"),
        ("mean_planner_latency_ms", "Planner latency (ms)"),
    ]
    for axis, (key, title) in zip(axes, compute_metrics):
        values = [summary["methods"][method].get(key) or 0.0 for method in methods]
        bars = axis.bar(labels, values, color="#4a5568")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.2f", fontsize=8)
    compute_path = output / "planner_compute.png"
    fig.savefig(compute_path, dpi=180)
    plt.close(fig)
    return [quality_path, compute_path]


def write_summary(summary: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
