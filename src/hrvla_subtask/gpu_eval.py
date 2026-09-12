"""GPU evaluation of real Cosmos-Reason2 proposals at shared task decision points."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

from .backends import CosmosReasonBackend
from .model import ExecutionMemory, TaskSpec
from .planner import PlannerConfig, WorldModelGuidedPlanner


def decision_points(task: TaskSpec, limit: int | None = None) -> list[frozenset[str]]:
    """Generate label-bearing states from the declared valid task trajectory."""
    output: list[frozenset[str]] = []
    state = task.initial_state
    while not task.complete(state) and (limit is None or len(output) < limit):
        output.append(state)
        skill = task.canonical_next(state)
        if skill is None:
            break
        state = skill.apply(state)
    return output


def evaluate_cosmos(
    tasks: Iterable[TaskSpec],
    *,
    model_path: str | Path,
    methods: list[str],
    max_points_per_task: int | None = None,
    seed: int = 42,
    config_overrides: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    backend = CosmosReasonBackend(model_path)
    records: list[dict[str, Any]] = []
    overrides = config_overrides or {}
    for method in methods:
        planner = WorldModelGuidedPlanner(backend, PlannerConfig(method=method, **overrides))
        for task_index, task in enumerate(tasks):
            memory = ExecutionMemory()
            for point_index, state in enumerate(decision_points(task, max_points_per_task)):
                canonical = task.canonical_next(state)
                before_tokens = backend.generated_tokens
                decision = planner.decide(
                    task,
                    state,
                    memory,
                    seed=seed + task_index * 1_000 + point_index,
                )
                selected = decision.selected.skill_id if decision.selected else None
                records.append(
                    {
                        "method": method,
                        "task_id": task.task_id,
                        "point_index": point_index,
                        "state": sorted(state),
                        "ground_truth": canonical.skill_id if canonical else None,
                        "selected": selected,
                        "correct": bool(canonical and selected == canonical.skill_id),
                        "valid": bool(
                            selected
                            and selected in task.skill_map
                            and task.skill_map[selected].applicable(state)
                        ),
                        "confidence": decision.confidence,
                        "route": decision.route,
                        "memory_update": decision.memory_update,
                        "safety_fallback": decision.safety_fallback,
                        "search_nodes": decision.search_nodes,
                        "model_calls": decision.model_calls,
                        "generated_tokens": backend.generated_tokens - before_tokens,
                        "latency_ms": decision.latency_ms,
                        "predicted_path": list(decision.predicted_path),
                        "candidates": [
                            {
                                "skill_id": item.skill_id,
                                "confidence": item.confidence,
                                "rationale": item.rationale,
                                "raw_text": item.raw_text,
                            }
                            for item in decision.candidates
                        ],
                    }
                )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["method"]].append(record)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "backend": "nvidia/Cosmos-Reason2-2B",
        "model_path": str(model_path),
        "peak_vram_gib": backend.peak_vram_bytes / (1024**3),
        "methods": {},
    }
    for method, rows in sorted(grouped.items()):
        latencies = sorted(float(row["latency_ms"]) for row in rows)
        p95 = latencies[max(0, min(len(latencies) - 1, round(0.95 * len(latencies)) - 1))]
        summary["methods"][method] = {
            "decision_points": len(rows),
            "next_subtask_accuracy": statistics.fmean(bool(row["correct"]) for row in rows),
            "valid_selection_rate": statistics.fmean(bool(row["valid"]) for row in rows),
            "safety_fallback_rate": statistics.fmean(
                bool(row["safety_fallback"]) for row in rows
            ),
            "ttc_route_rate": statistics.fmean(str(row["route"]).startswith("ttc") for row in rows),
            "mean_confidence": statistics.fmean(float(row["confidence"]) for row in rows),
            "mean_latency_ms": statistics.fmean(latencies),
            "p95_latency_ms": p95,
            "mean_search_nodes": statistics.fmean(int(row["search_nodes"]) for row in rows),
            "mean_model_calls": statistics.fmean(int(row["model_calls"]) for row in rows),
            "generated_tokens": sum(int(row["generated_tokens"]) for row in rows),
        }
    return records, summary


def write_gpu_results(
    records: list[dict[str, Any]], summary: dict[str, Any], output_dir: str | Path
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "cosmos_decisions.jsonl"
    raw_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary_path = output / "cosmos_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return [raw_path, summary_path]

    methods = list(summary["methods"])
    labels = [method.replace("_", "\n") for method in methods]
    accuracy = [summary["methods"][method]["next_subtask_accuracy"] for method in methods]
    latency = [summary["methods"][method]["mean_latency_ms"] for method in methods]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    bars = axes[0].bar(labels, accuracy, color="#2b6cb0")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Cosmos next-subtask accuracy")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].bar_label(bars, fmt="%.3f")
    bars = axes[1].bar(labels, latency, color="#4a5568")
    axes[1].set_title("End-to-end planner latency (ms)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].bar_label(bars, fmt="%.1f")
    chart_path = output / "cosmos_accuracy_latency.png"
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)
    return [raw_path, summary_path, chart_path]
