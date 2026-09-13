"""Transition-aware recovery contracts for long-horizon GR00T subtasks.

The implementation is inspired by BATON's invocation, handoff, and lookahead
transitions.  It is deliberately dependency-free and keeps recovery above the
unchanged GR00T/SONIC low-level action interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Iterable

from .model import FailureInjection, TaskSpec


RECOVERY_METHODS = (
    "no_recovery",
    "inner_monologue",
    "doremi",
    "rekep",
    "agentchord",
    "baton_str",
)


@dataclass(frozen=True)
class RecoveryPrimitive:
    """One bounded language-conditioned corrective action."""

    primitive_id: str
    instruction: str
    requires: frozenset[str]
    adds: frozenset[str]
    deletes: frozenset[str]
    cost: float
    risk: float
    success_probability: float
    precompiled: bool = False

    @property
    def milestone(self) -> str:
        return f"recovery:{self.primitive_id}"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RecoveryPrimitive":
        primitive = cls(
            primitive_id=str(raw["id"]),
            instruction=str(raw["instruction"]),
            requires=frozenset(map(str, raw.get("requires", []))),
            adds=frozenset(map(str, raw.get("adds", []))),
            deletes=frozenset(map(str, raw.get("deletes", []))),
            cost=float(raw.get("cost", 1.0)),
            risk=float(raw.get("risk", 0.0)),
            success_probability=float(raw.get("success_probability", 1.0)),
            precompiled=bool(raw.get("precompiled", False)),
        )
        if not primitive.primitive_id or not primitive.instruction:
            raise ValueError("recovery primitive id and instruction must be non-empty")
        if primitive.requires & primitive.adds:
            raise ValueError(f"{primitive.primitive_id}: requires overlaps adds")
        if primitive.adds & primitive.deletes:
            raise ValueError(f"{primitive.primitive_id}: adds overlaps deletes")
        if not 0.0 <= primitive.success_probability <= 1.0:
            raise ValueError(f"{primitive.primitive_id}: invalid success_probability")
        return primitive

    def applicable(self, state: Iterable[str]) -> bool:
        facts = set(state)
        return self.requires <= facts and self.milestone not in facts

    def apply(self, state: Iterable[str]) -> frozenset[str]:
        facts = set(state)
        if not self.applicable(facts):
            return frozenset(facts)
        facts.difference_update(self.deletes)
        facts.update(self.adds)
        facts.add(self.milestone)
        return frozenset(facts)


@dataclass(frozen=True)
class RecoveryScenario:
    """Fault model and transition contract for one task."""

    task_id: str
    failure_label: str
    failed_skill_id: str
    handoff_requires: frozenset[str]
    handoff_failure_probability: float
    recovery_primitives: tuple[RecoveryPrimitive, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RecoveryScenario":
        scenario = cls(
            task_id=str(raw["task_id"]),
            failure_label=str(raw["failure_label"]),
            failed_skill_id=str(raw["failed_skill_id"]),
            handoff_requires=frozenset(map(str, raw.get("handoff_requires", []))),
            handoff_failure_probability=float(raw.get("handoff_failure_probability", 0.0)),
            recovery_primitives=tuple(
                RecoveryPrimitive.from_dict(item) for item in raw.get("recovery_primitives", [])
            ),
        )
        if not 0.0 <= scenario.handoff_failure_probability <= 1.0:
            raise ValueError(f"{scenario.task_id}: invalid handoff failure probability")
        ids = [item.primitive_id for item in scenario.recovery_primitives]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{scenario.task_id}: recovery primitives must be non-empty and unique")
        return scenario

    @property
    def primitive_map(self) -> dict[str, RecoveryPrimitive]:
        return {item.primitive_id: item for item in self.recovery_primitives}

    def contract_satisfied(self, state: Iterable[str]) -> bool:
        return self.handoff_requires <= set(state)

    def contract_fraction(self, state: Iterable[str]) -> float:
        if not self.handoff_requires:
            return 1.0
        facts = set(state)
        return len(self.handoff_requires & facts) / len(self.handoff_requires)


def load_recovery_protocol(
    path: str | Path, tasks: Iterable[TaskSpec]
) -> dict[str, RecoveryScenario]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("recovery protocol schema_version must be 1")
    scenarios = [RecoveryScenario.from_dict(item) for item in payload.get("scenarios", [])]
    output = {item.task_id: item for item in scenarios}
    if len(output) != len(scenarios):
        raise ValueError("recovery protocol contains duplicate task_id")
    task_map = {task.task_id: task for task in tasks}
    if set(output) != set(task_map):
        raise ValueError("recovery protocol task IDs must exactly match the task suite")
    for task_id, scenario in output.items():
        task = task_map[task_id]
        matching = [
            item
            for item in task.failures
            if item.skill_id == scenario.failed_skill_id and item.label == scenario.failure_label
        ]
        if len(matching) != 1:
            raise ValueError(f"{task_id}: scenario must match exactly one declared failure")
    return output


@dataclass
class TransitionStatistics:
    successes: int = 0
    failures: int = 0
    total_cost: float = 0.0

    @property
    def posterior_success(self) -> float:
        return (self.successes + 1.0) / (self.successes + self.failures + 2.0)


@dataclass
class TransitionAwareMemory:
    """Auditable edge memory keyed by fault, skill, successor, and repair."""

    entries: dict[str, TransitionStatistics] = field(default_factory=dict)

    @staticmethod
    def key(
        scenario: RecoveryScenario,
        primitive_id: str,
        successor_id: str | None,
    ) -> str:
        return "|".join(
            (scenario.failure_label, scenario.failed_skill_id, successor_id or "goal", primitive_id)
        )

    def posterior(
        self,
        scenario: RecoveryScenario,
        primitive_id: str,
        successor_id: str | None,
    ) -> tuple[float, bool]:
        key = self.key(scenario, primitive_id, successor_id)
        stats = self.entries.get(key)
        return (stats.posterior_success, True) if stats else (0.5, False)

    def update(
        self,
        scenario: RecoveryScenario,
        primitive_id: str,
        successor_id: str | None,
        *,
        success: bool,
        cost: float,
    ) -> None:
        key = self.key(scenario, primitive_id, successor_id)
        stats = self.entries.setdefault(key, TransitionStatistics())
        stats.successes += int(success)
        stats.failures += int(not success)
        stats.total_cost += cost

    def to_dict(self) -> dict[str, Any]:
        return {key: asdict(value) for key, value in sorted(self.entries.items())}


@dataclass(frozen=True)
class FailureContext:
    task_id: str
    failure_label: str
    failed_skill_id: str
    state_before: frozenset[str]
    observed_state: frozenset[str]
    attempt: int


@dataclass(frozen=True)
class RecoveryDecision:
    method: str
    route: str
    primitive_id: str | None
    rollback: bool
    score: float
    contract_predicted: bool
    memory_hit: bool
    candidates_considered: int
    monitor_queries: int
    model_calls: int
    latency_ms: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TransitionRecoveryCoordinator:
    """Paper-aligned recovery policies over one shared corrective action catalog."""

    def __init__(self, method: str, memory: TransitionAwareMemory | None = None) -> None:
        if method not in RECOVERY_METHODS:
            raise ValueError(f"unknown recovery method: {method}")
        self.method = method
        self.memory = memory or TransitionAwareMemory()

    @staticmethod
    def _successor(task: TaskSpec, state: frozenset[str], failed_skill_id: str) -> str | None:
        failed = task.skill_map.get(failed_skill_id)
        if failed and failed.applicable(state):
            return failed_skill_id
        canonical = task.canonical_next(state)
        return canonical.skill_id if canonical else None

    def decide(
        self,
        task: TaskSpec,
        scenario: RecoveryScenario,
        failure: FailureInjection,
        context: FailureContext,
    ) -> RecoveryDecision:
        started = time.perf_counter()
        candidates = [
            item for item in scenario.recovery_primitives if item.applicable(context.observed_state)
        ]
        successor_id = self._successor(task, context.observed_state, context.failed_skill_id)
        primitive: RecoveryPrimitive | None = None
        rollback = False
        score = 0.0
        memory_hit = False
        route = self.method
        rationale = ""
        monitor_queries = 0
        model_calls = 0

        if self.method == "no_recovery":
            rationale = "leave the fault to the nominal closed-loop planner"
        elif self.method == "inner_monologue":
            monitor_queries = 1
            model_calls = 1
            primitive = min(candidates, key=lambda item: (item.cost, item.risk), default=None)
            rationale = "post-subgoal diagnosis followed by the cheapest valid correction"
        elif self.method == "doremi":
            monitor_queries = 3
            model_calls = 1
            primitive = max(
                candidates,
                key=lambda item: (item.success_probability, -item.cost),
                default=None,
            )
            rationale = "frequent monitor selects the highest immediate correction likelihood"
        elif self.method == "rekep":
            monitor_queries = 1
            rollback = True
            score = 1.0
            rationale = "restore the verified pre-failure checkpoint and retry"
        elif self.method == "agentchord":
            monitor_queries = 1
            precompiled = [item for item in candidates if item.precompiled]
            primitive = min(precompiled or candidates, key=lambda item: item.cost, default=None)
            rationale = "take the precompiled forward recovery edge"
        else:
            monitor_queries = 1
            route = "baton_str:invoke+handoff+lookahead"
            ranked: list[tuple[float, RecoveryPrimitive, bool]] = []
            for item in candidates:
                predicted = item.apply(context.observed_state)
                contract = scenario.contract_satisfied(predicted)
                posterior, hit = self.memory.posterior(scenario, item.primitive_id, successor_id)
                # A Beta posterior augments declared short-horizon reliability.  The
                # handoff term intentionally dominates local action cost: a locally
                # successful repair is useless if the failed/successor skill cannot inherit it.
                expected = 0.55 * item.success_probability + 0.45 * posterior
                candidate_score = (
                    5.0 * float(contract)
                    + 3.0 * expected
                    - 0.18 * item.cost
                    - 0.9 * item.risk
                    + 0.4 * float(bool(failure.recovered_when & predicted))
                )
                ranked.append((candidate_score, item, hit))
            if ranked:
                score, primitive, memory_hit = max(
                    ranked,
                    key=lambda row: (row[0], row[1].success_probability, -row[1].cost),
                )
                rationale = (
                    "select a readiness-valid repair whose exit satisfies the handoff "
                    "contract and remains usable by the failed/successor subtask"
                )
            else:
                rollback = True
                route += "+checkpoint"
                rationale = "no readiness-valid forward repair; use the last verified checkpoint"

        predicted_contract = bool(
            rollback
            or (
                primitive is not None
                and scenario.contract_satisfied(primitive.apply(context.observed_state))
            )
        )
        return RecoveryDecision(
            method=self.method,
            route=route,
            primitive_id=primitive.primitive_id if primitive else None,
            rollback=rollback,
            score=score,
            contract_predicted=predicted_contract,
            memory_hit=memory_hit,
            candidates_considered=len(candidates),
            monitor_queries=monitor_queries,
            model_calls=model_calls,
            latency_ms=(time.perf_counter() - started) * 1000,
            rationale=rationale,
        )
