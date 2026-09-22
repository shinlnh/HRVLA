"""Observed-state-only instruction routing for the GR00T semantic-action server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .backends import HeuristicProposalBackend
from .model import ExecutionMemory, TaskSpec
from .planner import PlannerConfig, WorldModelGuidedPlanner


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
