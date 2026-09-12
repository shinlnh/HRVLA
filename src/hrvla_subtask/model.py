"""Typed, dependency-free planning contracts.

The high-level planner deliberately communicates with GR00T through bounded natural-language
subtasks.  Continuous SONIC latent actions remain owned by the low-level policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class Skill:
    """One executable language-conditioned capability and its symbolic envelope."""

    skill_id: str
    instruction: str
    requires: frozenset[str] = frozenset()
    adds: frozenset[str] = frozenset()
    deletes: frozenset[str] = frozenset()
    cost: float = 1.0
    risk: float = 0.0
    priority: int = 0
    success_probability: float = 1.0
    terminal: bool = False
    required: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Skill":
        return cls(
            skill_id=str(raw["id"]),
            instruction=str(raw["instruction"]),
            requires=frozenset(map(str, raw.get("requires", []))),
            adds=frozenset(map(str, raw.get("adds", []))),
            deletes=frozenset(map(str, raw.get("deletes", []))),
            cost=float(raw.get("cost", 1.0)),
            risk=float(raw.get("risk", 0.0)),
            priority=int(raw.get("priority", 0)),
            success_probability=float(raw.get("success_probability", 1.0)),
            terminal=bool(raw.get("terminal", False)),
            required=bool(raw.get("required", True)),
        )

    @property
    def milestone(self) -> str:
        return f"milestone:{self.skill_id}"

    def applicable(self, state: Iterable[str]) -> bool:
        facts = set(state)
        return self.requires <= facts

    def useful(self, state: Iterable[str]) -> bool:
        facts = set(state)
        return self.milestone not in facts

    def apply(self, state: Iterable[str]) -> frozenset[str]:
        facts = set(state)
        if not self.applicable(facts):
            return frozenset(facts)
        facts.difference_update(self.deletes)
        facts.update(self.adds)
        facts.add(self.milestone)
        return frozenset(facts)


@dataclass(frozen=True)
class FailureInjection:
    """Deterministic failure injected on a particular attempt of a skill."""

    skill_id: str
    attempt: int = 1
    adds: frozenset[str] = frozenset()
    deletes: frozenset[str] = frozenset()
    recovered_when: frozenset[str] = frozenset()
    label: str = "execution_failure"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FailureInjection":
        return cls(
            skill_id=str(raw["skill_id"]),
            attempt=int(raw.get("attempt", 1)),
            adds=frozenset(map(str, raw.get("adds", []))),
            deletes=frozenset(map(str, raw.get("deletes", []))),
            recovered_when=frozenset(map(str, raw.get("recovered_when", []))),
            label=str(raw.get("label", "execution_failure")),
        )

    def recovered(self, state: Iterable[str]) -> bool:
        """Return whether observation facts prove recovery from this injection."""
        facts = set(state)
        if self.recovered_when:
            return self.recovered_when <= facts
        return bool(self.adds) and self.adds.isdisjoint(facts)


@dataclass(frozen=True)
class TaskSpec:
    """A reproducible long-horizon task and its available low-level skills."""

    task_id: str
    goal_instruction: str
    initial_state: frozenset[str]
    goal_state: frozenset[str]
    skills: tuple[Skill, ...]
    max_steps: int
    failures: tuple[FailureInjection, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TaskSpec":
        task = cls(
            task_id=str(raw["task_id"]),
            goal_instruction=str(raw["goal_instruction"]),
            initial_state=frozenset(map(str, raw.get("initial_state", []))),
            goal_state=frozenset(map(str, raw["goal_state"])),
            skills=tuple(Skill.from_dict(item) for item in raw["skills"]),
            max_steps=int(raw.get("max_steps", 32)),
            failures=tuple(FailureInjection.from_dict(item) for item in raw.get("failures", [])),
        )
        task.validate()
        return task

    @property
    def skill_map(self) -> dict[str, Skill]:
        return {skill.skill_id: skill for skill in self.skills}

    def validate(self) -> None:
        if not self.task_id or not self.goal_instruction:
            raise ValueError("task_id and goal_instruction must be non-empty")
        ids = [skill.skill_id for skill in self.skills]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.task_id}: duplicate skill id")
        if self.max_steps < 1:
            raise ValueError(f"{self.task_id}: max_steps must be positive")
        for skill in self.skills:
            if not 0.0 <= skill.success_probability <= 1.0:
                raise ValueError(f"{self.task_id}/{skill.skill_id}: invalid success_probability")
            if skill.requires & skill.adds:
                raise ValueError(f"{self.task_id}/{skill.skill_id}: requires overlaps adds")
            if skill.adds & skill.deletes:
                raise ValueError(f"{self.task_id}/{skill.skill_id}: adds overlaps deletes")
        unknown = {failure.skill_id for failure in self.failures} - set(ids)
        if unknown:
            raise ValueError(f"{self.task_id}: failure targets unknown skills: {sorted(unknown)}")
        if not self.reachable():
            raise ValueError(f"{self.task_id}: goal is not reachable from initial state")

    def complete(self, state: Iterable[str]) -> bool:
        return self.goal_state <= set(state)

    def progress(self, state: Iterable[str]) -> float:
        facts = set(state)
        milestones = [skill.milestone for skill in self.skills if skill.required]
        if not milestones:
            return float(self.complete(facts))
        return sum(item in facts for item in milestones) / len(milestones)

    def canonical_next(self, state: Iterable[str]) -> Skill | None:
        facts = set(state)
        choices = [
            skill
            for skill in self.skills
            if skill.applicable(facts) and skill.useful(facts) and not skill.terminal
        ]
        if not choices:
            return None
        return min(choices, key=lambda item: (item.priority, item.cost, item.skill_id))

    def reachable(self) -> bool:
        state = self.initial_state
        for _ in range(max(self.max_steps, len(self.skills) * 2)):
            if self.complete(state):
                return True
            applicable = [skill for skill in self.skills if skill.applicable(state) and skill.useful(state)]
            if not applicable:
                return False
            next_state = state
            for skill in applicable:
                next_state = skill.apply(next_state)
            if next_state == state:
                return False
            state = next_state
        return self.complete(state)


@dataclass(frozen=True)
class Candidate:
    skill_id: str
    confidence: float
    rationale: str = ""
    raw_text: str = ""

    def normalized(self) -> "Candidate":
        return Candidate(
            skill_id=self.skill_id.strip(),
            confidence=max(0.0, min(1.0, float(self.confidence))),
            rationale=self.rationale.strip(),
            raw_text=self.raw_text,
        )


@dataclass
class ExecutionMemory:
    """Observation-aligned task memory; never treats a prediction as an outcome."""

    completed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    last_subtask: str | None = None
    revisions: int = 0
    observations: int = 0

    def clone(self) -> "ExecutionMemory":
        return ExecutionMemory(
            completed=list(self.completed),
            failed=list(self.failed),
            last_subtask=self.last_subtask,
            revisions=self.revisions,
            observations=self.observations,
        )

    def reconcile(self, task: TaskSpec, state: Iterable[str]) -> str:
        """Catch up or roll back memory using observed persistent milestones."""
        facts = set(state)
        observed = [skill.skill_id for skill in task.skills if skill.milestone in facts]
        previous = list(self.completed)
        self.completed = observed
        self.observations += 1
        if previous == observed:
            return "aligned"
        self.revisions += 1
        if set(previous) <= set(observed):
            return "catch_up"
        if set(observed) < set(previous):
            return "rollback"
        return "repair"

    def mark_failure(self, skill_id: str) -> None:
        self.failed.append(skill_id)
        self.last_subtask = skill_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchBranch:
    state: frozenset[str]
    path: tuple[str, ...]
    score: float
    memory: ExecutionMemory


@dataclass(frozen=True)
class Decision:
    selected: Candidate | None
    route: str
    confidence: float
    memory_update: str
    candidates: tuple[Candidate, ...]
    search_nodes: int
    model_calls: int
    latency_ms: float
    predicted_path: tuple[str, ...] = ()
    safety_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        return raw
