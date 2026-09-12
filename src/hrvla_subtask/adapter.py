"""Thin language adapter between a high-level planner and Isaac-GR00T PolicyClient."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .model import Decision, ExecutionMemory, TaskSpec
from .planner import WorldModelGuidedPlanner


class ActionPolicy(Protocol):
    def get_action(self, observation: dict[str, Any]) -> Any: ...


StateExtractor = Callable[[dict[str, Any]], frozenset[str]]


@dataclass
class Gr00tSubtaskAdapter:
    """Replace only GR00T's language field, leaving video/state/action schemas untouched.

    The caller owns perception and supplies ``state_extractor``.  This prevents the planner from
    pretending that symbolic predicates were observed when no detector produced them.
    """

    policy: ActionPolicy
    planner: WorldModelGuidedPlanner
    task: TaskSpec
    state_extractor: StateExtractor
    language_key: str = "annotation.human.task_description"
    memory: ExecutionMemory = field(default_factory=ExecutionMemory)
    seed: int = 0
    last_decision: Decision | None = None

    def get_action(self, observation: dict[str, Any]) -> Any:
        state = self.state_extractor(observation)
        self.last_decision = self.planner.decide(
            self.task, state, self.memory, seed=self.seed + self.memory.observations
        )
        if self.last_decision.selected is None:
            raise RuntimeError("task is complete; no further GR00T action should be requested")
        skill = self.task.skill_map[self.last_decision.selected.skill_id]
        planned_observation = deepcopy(observation)
        planned_observation.setdefault("language", {})[self.language_key] = [[skill.instruction]]
        return self.policy.get_action(planned_observation)

    def mark_execution_failure(self) -> None:
        if self.last_decision and self.last_decision.selected:
            self.memory.mark_failure(self.last_decision.selected.skill_id)
