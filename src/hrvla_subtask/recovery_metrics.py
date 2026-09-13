"""Metrics and figures for paired-seed subtask recovery evaluation."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float] | None:
    if not trials:
        return None
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def summarize_recovery(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["method"])].append(record)
    output: dict[str, Any] = {"schema_version": 1, "methods": {}}
    for method, rows in sorted(grouped.items()):
        episodes = len(rows)
        successes = sum(bool(row["success"]) for row in rows)
        failures = sum(int(row["failures"]) for row in rows)
        recovered = sum(int(row["recovered_failures"]) for row in rows)
        handoff_checks = sum(int(row["handoff_checks"]) for row in rows)
        contracts = sum(float(row["contract_satisfied"]) for row in rows)
        recovery_decisions = sum(int(row["recovery_decisions"]) for row in rows)
        latencies = [float(row["planner_latency_ms"]) for row in rows]
        output["methods"][method] = {
            "episodes": episodes,
            "task_success_rate": successes / episodes,
            "task_success_wilson95": _wilson(successes, episodes),
            "mean_progress": statistics.fmean(float(row["progress"]) for row in rows),
            "failure_recovery_rate": _ratio(recovered, failures),
            "failure_recovery_wilson95": _wilson(recovered, failures),
            "handoff_contract_rate": _ratio(contracts, handoff_checks),
            "handoff_failure_rate": _ratio(
                sum(int(row["handoff_failures"]) for row in rows), handoff_checks
            ),
            "mean_actions": statistics.fmean(int(row["actions"]) for row in rows),
            "recovery_actions_per_failure": _ratio(
                sum(int(row["recovery_actions"]) for row in rows), failures
            ),
            "mean_recovery_latency_actions": _ratio(
                sum(int(row["recovery_latency_actions"]) for row in rows), recovered
            ),
            "rollback_actions_per_episode": statistics.fmean(
                int(row["rollback_actions"]) for row in rows
            ),
            "monitor_queries_per_failure": _ratio(
                sum(int(row["monitor_queries"]) for row in rows), failures
            ),
            "recovery_model_calls_per_failure": _ratio(
                sum(int(row["recovery_model_calls"]) for row in rows), failures
            ),
            "nominal_model_calls_per_episode": statistics.fmean(
                int(row["nominal_model_calls"]) for row in rows
            ),
            "mean_planner_latency_ms": statistics.fmean(latencies),
            "mean_recovery_decision_latency_ms": _ratio(
                sum(float(row["recovery_latency_ms"]) for row in rows), recovery_decisions
            ),
            "memory_hit_rate": _ratio(
                sum(int(row["memory_hits"]) for row in rows), recovery_decisions
            ),
            "escalations_per_episode": statistics.fmean(int(row["escalations"]) for row in rows),
        }
    return output


def write_recovery_summary(summary: dict[str, Any], output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    methods = summary["methods"]
    scalar_keys = sorted(
        {
            key
            for values in methods.values()
            for key, value in values.items()
            if key != "episodes" and not isinstance(value, list)
        }
    )
    csv_path = output / "chart_data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "episodes", *scalar_keys])
        for method, values in methods.items():
            writer.writerow([method, values["episodes"], *(values.get(key) for key in scalar_keys)])
    return [summary_path, csv_path]


def plot_recovery_summary(summary: dict[str, Any], output_dir: str | Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required to render recovery charts") from exc

    output = Path(output_dir)
    methods = list(summary["methods"])
    labels = [item.replace("_", "\n") for item in methods]
    colors = ["#718096", "#805ad5", "#d69e2e", "#c53030", "#319795", "#2b6cb0"]

    quality = [
        ("task_success_rate", "Task success"),
        ("failure_recovery_rate", "Failure recovery"),
        ("handoff_contract_rate", "Handoff contract"),
        ("mean_progress", "Required progress"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, quality):
        values = [summary["methods"][method].get(key) or 0.0 for method in methods]
        bars = axis.bar(labels, values, color=colors[: len(methods)])
        axis.set_ylim(0.0, 1.05)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.3f", fontsize=7)
    quality_path = output / "recovery_quality.png"
    fig.savefig(quality_path, dpi=180)
    plt.close(fig)

    efficiency = [
        ("mean_actions", "Episode actions"),
        ("mean_recovery_latency_actions", "Actions to recovery"),
        ("monitor_queries_per_failure", "Monitor queries / failure"),
        ("handoff_failure_rate", "Handoff failure rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, efficiency):
        values = [summary["methods"][method].get(key) or 0.0 for method in methods]
        bars = axis.bar(labels, values, color=colors[: len(methods)])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.2f", fontsize=7)
    efficiency_path = output / "recovery_efficiency.png"
    fig.savefig(efficiency_path, dpi=180)
    plt.close(fig)
    return [quality_path, efficiency_path]
