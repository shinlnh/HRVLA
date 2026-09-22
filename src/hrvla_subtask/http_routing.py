"""Observed-state-only instruction routing for the GR00T semantic-action server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .backends import HeuristicProposalBackend
from .model import ExecutionMemory, TaskSpec
from .planner import PlannerConfig, WorldModelGuidedPlanner
from .recovery import (
    FailureContext,
    TransitionRecoveryCoordinator,
    load_recovery_protocol,
)


class ObservedStateRouter:
    """Plan subtasks without inferring unobserved simulator predicates."""

    def __init__(self, program: Mapping[str, Any], *, seed: int = 0) -> None:
        if program.get("schema_version") != 1:
            raise ValueError("subtask program schema_version must be 1")
        tasks = [TaskSpec.from_dict(row) for row in program.get("tasks", [])]
        if not tasks or len({task.task_id for task in tasks}) != len(tasks):
            raise ValueError("subtask program must contain unique tasks")
        self.tasks = {task.task_id: task for task in tasks}
        self.seed = int(seed)
        self.planner = WorldModelGuidedPlanner(
            HeuristicProposalBackend(error_rate=0.0), PlannerConfig()
        )
        self.reset()

    @classmethod
    def from_path(cls, path: Path, *, seed: int = 0) -> "ObservedStateRouter":
        return cls(json.loads(path.read_text(encoding="utf-8")), seed=seed)

    def reset(self) -> None:
        self.memories = {task_id: ExecutionMemory() for task_id in self.tasks}
        self.decisions = {task_id: 0 for task_id in self.tasks}

    def select(self, task_id: str | None, payload: Mapping[str, Any]) -> tuple[str, str]:
        if task_id not in self.tasks:
            raise ValueError(f"no declared subtask program for task: {task_id}")
        raw_state = payload.get("observed_predicates")
        if not isinstance(raw_state, list) or any(not isinstance(x, str) for x in raw_state):
            raise ValueError("observed_predicates must be a list of detector-produced strings")
        task = self.tasks[task_id]
        state = frozenset(raw_state)
        declared = set(task.initial_state) | set(task.goal_state)
        for skill in task.skills:
            declared.update(skill.requires | skill.adds | skill.deletes | {skill.milestone})
        if not state <= declared:
            raise ValueError(f"unknown observed predicates: {sorted(state - declared)}")
        decision = self.planner.decide(
            task, state, self.memories[task_id], seed=self.seed + self.decisions[task_id]
        )
        if decision.selected is None:
            raise ValueError("subtask program has no executable skill for observed state")
        self.decisions[task_id] += 1
        skill = task.skill_map[decision.selected.skill_id]
        return skill.instruction, skill.skill_id


class RecoveryObservedStateRouter:
    """Select a bounded repair only for an explicitly observed active failure."""

    def __init__(self, program: Mapping[str, Any], protocol_path: Path, *, seed: int = 0) -> None:
        self.subtask = ObservedStateRouter(program, seed=seed)
        self.scenarios = load_recovery_protocol(protocol_path, self.subtask.tasks.values())
        self.coordinators = {
            task_id: TransitionRecoveryCoordinator("baton_str") for task_id in self.scenarios
        }

    @classmethod
    def from_paths(
        cls, program_path: Path, protocol_path: Path, *, seed: int = 0
    ) -> "RecoveryObservedStateRouter":
        return cls(json.loads(program_path.read_text(encoding="utf-8")), protocol_path, seed=seed)

    def reset(self) -> None:
        self.subtask.reset()
        self.coordinators = {
            task_id: TransitionRecoveryCoordinator("baton_str") for task_id in self.scenarios
        }

    def select(self, task_id: str | None, payload: Mapping[str, Any]) -> tuple[str, str]:
        if payload.get("failure_active") is not True:
            return self.subtask.select(task_id, payload)
        if task_id not in self.scenarios:
            raise ValueError(f"no recovery scenario for task: {task_id}")
        scenario = self.scenarios[task_id]
        if payload.get("failure_label") != scenario.failure_label:
            raise ValueError("active failure label does not match the frozen scenario")
        observed = payload.get("observed_predicates")
        before = payload.get("pre_failure_predicates")
        if not isinstance(observed, list) or not all(isinstance(x, str) for x in observed):
            raise ValueError("recovery requires detector-produced observed_predicates")
        if not isinstance(before, list) or not all(isinstance(x, str) for x in before):
            raise ValueError("recovery requires detector-produced pre_failure_predicates")
        task = self.subtask.tasks[task_id]
        failure = next(
            item for item in task.failures
            if item.skill_id == scenario.failed_skill_id and item.label == scenario.failure_label
        )
        context = FailureContext(
            task_id=task_id,
            failure_label=scenario.failure_label,
            failed_skill_id=scenario.failed_skill_id,
            state_before=frozenset(before),
            observed_state=frozenset(observed),
            attempt=int(payload.get("failure_attempt", 1)),
        )
        decision = self.coordinators[task_id].decide(task, scenario, failure, context)
        if decision.rollback or decision.primitive_id is None:
            raise ValueError("recovery requires a verified checkpoint rollback; no action emitted")
        primitive = scenario.primitive_map[decision.primitive_id]
        return primitive.instruction, primitive.primitive_id
