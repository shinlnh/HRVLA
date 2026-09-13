"""Cosmos-Reason2 evaluation at observed subtask-failure decision points."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

from .backends import CosmosReasonBackend
from .model import ExecutionMemory, Skill, TaskSpec
from .recovery import (
    FailureContext,
    RecoveryScenario,
    TransitionRecoveryCoordinator,
)


GPU_RECOVERY_METHODS = ("cosmos_direct", "cosmos_best_of_n", "cosmos_baton_str")


def _augmented_task(task: TaskSpec, scenario: RecoveryScenario) -> TaskSpec:
    recovery_skills = tuple(
        Skill(
            skill_id=item.primitive_id,
            instruction=item.instruction,
            requires=item.requires,
            adds=item.adds,
            deletes=item.deletes,
            cost=item.cost,
            risk=item.risk,
            priority=5,
            success_probability=item.success_probability,
            required=False,
        )
        for item in scenario.recovery_primitives
    )
    return TaskSpec(
        task_id=task.task_id,
        goal_instruction=(
            f"Recover from {scenario.failure_label}. The recovery handoff must establish "
            f"{sorted(scenario.handoff_requires)} before continuing: {task.goal_instruction}"
        ),
        initial_state=task.initial_state,
        goal_state=task.goal_state,
        skills=task.skills + recovery_skills,
        max_steps=task.max_steps,
        failures=task.failures,
    )


def _failure_point(task: TaskSpec, scenario: RecoveryScenario) -> tuple[frozenset[str], frozenset[str]]:
    state = task.initial_state
    for _ in range(task.max_steps):
        skill = task.canonical_next(state)
        if skill is None:
            break
        if skill.skill_id == scenario.failed_skill_id:
            before = state
            failure = next(item for item in task.failures if item.label == scenario.failure_label)
            facts = set(state)
            facts.difference_update(failure.deletes)
            facts.update(failure.adds)
            return before, frozenset(facts)
        state = skill.apply(state)
    raise ValueError(f"{task.task_id}: failed skill is absent from canonical trajectory")


def _apply_candidate(
    task: TaskSpec,
    scenario: RecoveryScenario,
    state: frozenset[str],
    skill_id: str | None,
) -> tuple[bool, float]:
    if not skill_id:
        return False, 0.0
    primitive = scenario.primitive_map.get(skill_id)
    if primitive is not None:
        if not primitive.applicable(state):
            return False, 0.0
        predicted = primitive.apply(state)
        return True, scenario.contract_fraction(predicted)
    skill = task.skill_map.get(skill_id)
    if skill is None or not skill.applicable(state):
        return False, 0.0
    return True, scenario.contract_fraction(skill.apply(state))


def evaluate_cosmos_recovery(
    tasks: Iterable[TaskSpec],
    scenarios: dict[str, RecoveryScenario],
    *,
    model_path: str | Path,
    candidates_per_failure: int = 3,
    repetitions: int = 1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    backend = CosmosReasonBackend(model_path)
    records: list[dict[str, Any]] = []
    task_list = list(tasks)
    evaluation_points = (
        (repetition, task_index, task)
        for repetition in range(repetitions)
        for task_index, task in enumerate(task_list)
    )
    for repetition, task_index, task in evaluation_points:
        scenario = scenarios[task.task_id]
        augmented = _augmented_task(task, scenario)
        before, failed_state = _failure_point(task, scenario)
        memory = ExecutionMemory()
        memory.reconcile(task, before)
        started = time.perf_counter()
        tokens_before = backend.generated_tokens
        candidates = backend.propose(
            augmented,
            failed_state,
            memory,
            candidates_per_failure,
            seed + repetition * 100_003 + task_index * 1009,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        generated_tokens = backend.generated_tokens - tokens_before
        failure = next(item for item in task.failures if item.label == scenario.failure_label)
        context = FailureContext(
            task_id=task.task_id,
            failure_label=failure.label,
            failed_skill_id=failure.skill_id,
            state_before=before,
            observed_state=failed_state,
            attempt=1,
        )
        oracle = TransitionRecoveryCoordinator("baton_str").decide(
            task, scenario, failure, context
        )
        optimal_id = oracle.primitive_id

        direct = candidates[0] if candidates else None
        valid_candidates = [
            item for item in candidates if _apply_candidate(task, scenario, failed_state, item.skill_id)[0]
        ]
        best = max(valid_candidates, key=lambda item: item.confidence, default=direct)
        protocol_candidates = [
            item
            for item in valid_candidates
            if item.skill_id in scenario.primitive_map
        ]
        baton_fallback = not protocol_candidates
        if protocol_candidates:
            baton = max(
                protocol_candidates,
                key=lambda item: (
                    5.0
                    * scenario.contract_fraction(
                        scenario.primitive_map[item.skill_id].apply(failed_state)
                    )
                    + item.confidence
                    - 0.2 * scenario.primitive_map[item.skill_id].risk
                ),
            )
            baton_id = baton.skill_id
            baton_confidence = baton.confidence
        else:
            baton_id = optimal_id
            baton_confidence = 0.0

        selections = {
            "cosmos_direct": (direct.skill_id if direct else None, direct.confidence if direct else 0.0, False),
            "cosmos_best_of_n": (best.skill_id if best else None, best.confidence if best else 0.0, False),
            "cosmos_baton_str": (baton_id, baton_confidence, baton_fallback),
        }
        serialized_candidates = [
            {
                "skill_id": item.skill_id,
                "confidence": item.confidence,
                "rationale": item.rationale,
                "raw_text": item.raw_text,
            }
            for item in candidates
        ]
        for method, (selected, confidence, fallback) in selections.items():
            valid, contract_fraction = _apply_candidate(task, scenario, failed_state, selected)
            records.append(
                {
                    "method": method,
                    "repetition": repetition,
                    "task_id": task.task_id,
                    "failure_label": scenario.failure_label,
                    "failed_state": sorted(failed_state),
                    "handoff_requires": sorted(scenario.handoff_requires),
                    "selected": selected,
                    "optimal_transition_recovery": optimal_id,
                    "optimal_selection": selected == optimal_id,
                    "valid": valid,
                    "contract_fraction": contract_fraction,
                    "full_contract": contract_fraction == 1.0,
                    "confidence": confidence,
                    "safety_fallback": fallback,
                    "latency_ms": latency_ms,
                    "generated_tokens": generated_tokens,
                    "candidates": serialized_candidates,
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
        "shared_candidate_sets": True,
        "methods": {},
    }
    for method, rows in sorted(grouped.items()):
        latencies = sorted(float(row["latency_ms"]) for row in rows)
        summary["methods"][method] = {
            "decision_points": len(rows),
            "optimal_selection_rate": statistics.fmean(bool(row["optimal_selection"]) for row in rows),
            "valid_selection_rate": statistics.fmean(bool(row["valid"]) for row in rows),
            "full_handoff_contract_rate": statistics.fmean(bool(row["full_contract"]) for row in rows),
            "mean_handoff_contract_fraction": statistics.fmean(
                float(row["contract_fraction"]) for row in rows
            ),
            "safety_fallback_rate": statistics.fmean(bool(row["safety_fallback"]) for row in rows),
            "mean_confidence": statistics.fmean(float(row["confidence"]) for row in rows),
            "mean_latency_ms": statistics.fmean(latencies),
            "p95_latency_ms": latencies[max(0, round(0.95 * len(latencies)) - 1)],
            "generated_tokens": sum(int(row["generated_tokens"]) for row in rows),
        }
    return records, summary


def write_cosmos_recovery_results(
    records: list[dict[str, Any]], summary: dict[str, Any], output_dir: str | Path
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "cosmos_recovery_decisions.jsonl"
    raw_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records), encoding="utf-8"
    )
    summary_path = output / "cosmos_recovery_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return [raw_path, summary_path]

    methods = list(summary["methods"])
    labels = [item.replace("_", "\n") for item in methods]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    metrics = [
        ("optimal_selection_rate", "Optimal recovery"),
        ("full_handoff_contract_rate", "Full handoff contract"),
        ("safety_fallback_rate", "Safety fallback"),
    ]
    for axis, (key, title) in zip(axes, metrics):
        values = [summary["methods"][method][key] for method in methods]
        bars = axis.bar(labels, values, color=["#718096", "#d69e2e", "#2b6cb0"])
        axis.set_ylim(0, 1.05)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.3f", fontsize=8)
    chart = output / "cosmos_recovery_comparison.png"
    fig.savefig(chart, dpi=180)
    plt.close(fig)
    return [raw_path, summary_path, chart]
