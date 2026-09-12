"""Deterministic symbolic rollouts for high-level planner evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable

from .backends import HeuristicProposalBackend
from .model import ExecutionMemory, FailureInjection, TaskSpec
from .planner import PlannerConfig, WorldModelGuidedPlanner


def load_suite(path: str | Path) -> list[TaskSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("suite schema_version must be 1")
    tasks = [TaskSpec.from_dict(item) for item in payload.get("tasks", [])]
    if not tasks:
        raise ValueError("suite contains no tasks")
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("suite contains duplicate task_id")
    return tasks


@dataclass
class EpisodeResult:
    method: str
    task_id: str
    seed: int
    success: bool
    progress: float
    steps: int
    injected_failures: int
    recovered_failures: int
    correct_decisions: int
    decisions: int
    valid_selected: int
    safety_fallbacks: int
    hallucinated_candidates: int
    repeated_actions: int
    memory_revisions: int
    ttc_decisions: int
    search_nodes: int
    model_calls: int
    generated_tokens: int
    planner_latency_ms: float
    trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _inject(state: frozenset[str], failure: FailureInjection) -> frozenset[str]:
    facts = set(state)
    facts.difference_update(failure.deletes)
    facts.update(failure.adds)
    return frozenset(facts)


def run_episode(
    task: TaskSpec,
    *,
    method: str,
    seed: int,
    proposal_error_rate: float = 0.28,
    planner_overrides: dict[str, Any] | None = None,
) -> EpisodeResult:
    rng = random.Random(seed)
    state = task.initial_state
    memory = ExecutionMemory()
    backend = HeuristicProposalBackend(error_rate=proposal_error_rate)
    config_values: dict[str, Any] = {
        "method": method,
        "branching_factor": 4,
        "beam_width": 3,
        "search_depth": 3,
        "confidence_threshold": 0.72,
    }
    config_values.update(planner_overrides or {})
    planner = WorldModelGuidedPlanner(backend, PlannerConfig(**config_values))
    attempts: dict[str, int] = {}
    trace: list[dict[str, Any]] = []
    injected = 0
    recovered = 0
    active_injections: list[FailureInjection] = []
    correct = decisions = valid = fallbacks = hallucinations = repeats = 0
    ttc_decisions = search_nodes = model_calls = 0
    latency = 0.0
    previous: str | None = None

    for step in range(task.max_steps):
        if task.complete(state):
            break
        canonical = task.canonical_next(state)
        decision = planner.decide(task, state, memory, seed=seed * 1000 + step)
        decisions += 1
        latency += decision.latency_ms
        search_nodes += decision.search_nodes
        model_calls += decision.model_calls
        ttc_decisions += decision.route.startswith("ttc")
        fallbacks += decision.safety_fallback
        hallucinations += sum(item.skill_id not in task.skill_map for item in decision.candidates)
        selected_id = decision.selected.skill_id if decision.selected else "done"
        correct += bool(canonical and selected_id == canonical.skill_id)
        if previous == selected_id:
            repeats += 1
        previous = selected_id
        skill = task.skill_map.get(selected_id)
        is_valid = bool(skill and skill.applicable(state) and skill.useful(state))
        valid += is_valid
        event: dict[str, Any] = {
            "step": step,
            "state_before": sorted(state),
            "canonical": canonical.skill_id if canonical else None,
            "selected": selected_id,
            "route": decision.route,
            "confidence": decision.confidence,
            "predicted_path": list(decision.predicted_path),
            "valid": is_valid,
            "success": False,
        }
        if not is_valid or skill is None:
            memory.mark_failure(selected_id)
            event["outcome"] = "invalid"
            trace.append(event)
            continue

        attempts[selected_id] = attempts.get(selected_id, 0) + 1
        failure = next(
            (
                item
                for item in task.failures
                if item.skill_id == selected_id and item.attempt == attempts[selected_id]
            ),
            None,
        )
        if failure is not None:
            state = _inject(state, failure)
            memory.mark_failure(selected_id)
            injected += 1
            active_injections.append(failure)
            event.update({"outcome": "injected_failure", "failure": failure.label})
        elif rng.random() <= skill.success_probability:
            state = skill.apply(state)
            event.update({"outcome": "executed", "success": True})
        else:
            memory.mark_failure(selected_id)
            event["outcome"] = "stochastic_failure"
        newly_recovered = [item for item in active_injections if item.recovered(state)]
        if newly_recovered:
            recovered += len(newly_recovered)
            active_injections = [item for item in active_injections if not item.recovered(state)]
            event["recovered_failures"] = [item.label for item in newly_recovered]
        event["state_after"] = sorted(state)
        trace.append(event)

    return EpisodeResult(
        method=method,
        task_id=task.task_id,
        seed=seed,
        success=task.complete(state),
        progress=task.progress(state),
        steps=len(trace),
        injected_failures=injected,
        recovered_failures=recovered,
        correct_decisions=correct,
        decisions=decisions,
        valid_selected=valid,
        safety_fallbacks=fallbacks,
        hallucinated_candidates=hallucinations,
        repeated_actions=repeats,
        memory_revisions=memory.revisions,
        ttc_decisions=ttc_decisions,
        search_nodes=search_nodes,
        model_calls=model_calls,
        generated_tokens=backend.generated_tokens,
        planner_latency_ms=latency,
        trace=trace,
    )


def episode_job(raw_task: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pickle-safe process-pool entry point."""
    return run_episode(TaskSpec.from_dict(raw_task), **kwargs).to_dict()


def task_to_dict(task: TaskSpec) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "goal_instruction": task.goal_instruction,
        "initial_state": sorted(task.initial_state),
        "goal_state": sorted(task.goal_state),
        "max_steps": task.max_steps,
        "skills": [
            {
                "id": skill.skill_id,
                "instruction": skill.instruction,
                "requires": sorted(skill.requires),
                "adds": sorted(skill.adds),
                "deletes": sorted(skill.deletes),
                "cost": skill.cost,
                "risk": skill.risk,
                "priority": skill.priority,
                "success_probability": skill.success_probability,
                "terminal": skill.terminal,
                "required": skill.required,
            }
            for skill in task.skills
        ],
        "failures": [
            {
                "skill_id": item.skill_id,
                "attempt": item.attempt,
                "adds": sorted(item.adds),
                "deletes": sorted(item.deletes),
                "recovered_when": sorted(item.recovered_when),
                "label": item.label,
            }
            for item in task.failures
        ],
    }
