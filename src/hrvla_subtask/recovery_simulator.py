"""Fault-injected long-horizon simulation for subtask recovery policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any

from .backends import HeuristicProposalBackend
from .model import ExecutionMemory, FailureInjection, TaskSpec
from .planner import PlannerConfig, WorldModelGuidedPlanner
from .recovery import (
    FailureContext,
    RecoveryScenario,
    TransitionAwareMemory,
    TransitionRecoveryCoordinator,
)
from .simulator import task_to_dict


@dataclass
class RecoveryEpisodeResult:
    method: str
    task_id: str
    seed: int
    disturbance_rate: float
    success: bool
    progress: float
    actions: int
    failures: int
    recovered_failures: int
    recovery_actions: int
    recovery_decisions: int
    recovery_latency_actions: int
    rollback_actions: int
    handoff_checks: int
    handoff_failures: int
    contract_satisfied: float
    monitor_queries: int
    recovery_model_calls: int
    nominal_model_calls: int
    planner_latency_ms: float
    recovery_latency_ms: float
    memory_hits: int
    escalations: int
    trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _inject(state: frozenset[str], failure: FailureInjection) -> frozenset[str]:
    facts = set(state)
    facts.difference_update(failure.deletes)
    facts.update(failure.adds)
    return frozenset(facts)


def _find_failure(task: TaskSpec, scenario: RecoveryScenario) -> FailureInjection:
    return next(
        item
        for item in task.failures
        if item.skill_id == scenario.failed_skill_id and item.label == scenario.failure_label
    )


def run_recovery_episode(
    task: TaskSpec,
    scenario: RecoveryScenario,
    *,
    method: str,
    seed: int,
    disturbance_rate: float = 0.15,
    proposal_error_rate: float = 0.28,
    max_extra_steps: int = 16,
) -> RecoveryEpisodeResult:
    """Run one paired-seed episode with a guaranteed fault plus repeated disturbances."""
    rng = random.Random(seed)
    state = task.initial_state
    nominal_memory = ExecutionMemory()
    transition_memory = TransitionAwareMemory()
    coordinator = TransitionRecoveryCoordinator(method, transition_memory)
    backend = HeuristicProposalBackend(error_rate=proposal_error_rate)
    planner = WorldModelGuidedPlanner(
        backend,
        PlannerConfig(
            method="adaptive_ttc",
            branching_factor=4,
            beam_width=3,
            search_depth=3,
            confidence_threshold=0.72,
        ),
    )
    failure = _find_failure(task, scenario)
    attempts: dict[str, int] = {}
    trace: list[dict[str, Any]] = []
    forced_failure_pending = True
    active_started: int | None = None
    active_failure = False
    recovery_primitive_id: str | None = None
    skip_recovery_once = False
    failures = recovered = recovery_actions = recovery_decisions = 0
    actions_taken = 0
    recovery_latency_actions = rollback_actions = handoff_checks = handoff_failures = 0
    contract_satisfied = 0.0
    monitor_queries = recovery_model_calls = nominal_model_calls = 0
    planner_latency = recovery_latency = 0.0
    memory_hits = escalations = 0
    max_actions = task.max_steps + max_extra_steps

    for action_index in range(max_actions):
        if task.complete(state):
            break

        if active_failure and failure.recovered(state):
            active_failure = False
            recovered += 1
            recovery_latency_actions += action_index - (
                active_started if active_started is not None else action_index
            )
            if recovery_primitive_id is not None:
                transition_memory.update(
                    scenario,
                    recovery_primitive_id,
                    scenario.failed_skill_id,
                    success=True,
                    cost=float(action_index - (active_started or action_index)),
                )
            trace.append(
                {
                    "action": action_index,
                    "kind": "recovery_verified",
                    "failure": failure.label,
                    "state": sorted(state),
                }
            )

        applicable_recovery = any(
            item.applicable(state) for item in scenario.recovery_primitives
        )
        if active_failure and not skip_recovery_once and applicable_recovery:
            context = FailureContext(
                task_id=task.task_id,
                failure_label=failure.label,
                failed_skill_id=failure.skill_id,
                state_before=frozenset(),
                observed_state=state,
                attempt=attempts.get(failure.skill_id, 0),
            )
            recovery_decision = coordinator.decide(task, scenario, failure, context)
            recovery_decisions += 1
            monitor_queries += recovery_decision.monitor_queries
            recovery_model_calls += recovery_decision.model_calls
            recovery_latency += recovery_decision.latency_ms
            memory_hits += int(recovery_decision.memory_hit)

            if recovery_decision.rollback:
                # Backtracking rewinds the task graph/controller, not the physical
                # world.  A dropped object remains dropped; treating this as an
                # oracle state reset would give the baseline information it does not have.
                rollback_actions += 1
                recovery_actions += 1
                actions_taken += 1
                skip_recovery_once = True
                trace.append(
                    {
                        "action": action_index,
                        "kind": "controller_backtrack",
                        "route": recovery_decision.route,
                        "state_after": sorted(state),
                        "decision": recovery_decision.to_dict(),
                    }
                )
                continue
            primitive = scenario.primitive_map.get(recovery_decision.primitive_id or "")
            if primitive is not None and primitive.applicable(state):
                recovery_actions += 1
                actions_taken += 1
                before = state
                executed = rng.random() <= primitive.success_probability
                if executed:
                    state = primitive.apply(state)
                    recovery_primitive_id = primitive.primitive_id
                transition_memory.update(
                    scenario,
                    primitive.primitive_id,
                    scenario.failed_skill_id,
                    success=executed and scenario.contract_satisfied(state),
                    cost=primitive.cost,
                )
                trace.append(
                    {
                        "action": action_index,
                        "kind": "recovery_primitive",
                        "primitive": primitive.primitive_id,
                        "instruction": primitive.instruction,
                        "executed": executed,
                        "contract_satisfied": scenario.contract_satisfied(state),
                        "state_before": sorted(before),
                        "state_after": sorted(state),
                        "decision": recovery_decision.to_dict(),
                    }
                )
                continue
            if method != "no_recovery" and recovery_decision.primitive_id is None:
                escalations += 1

        skip_recovery_once = False
        state_before = state
        actions_taken += 1
        decision = planner.decide(task, state, nominal_memory, seed=seed * 10_000 + action_index)
        nominal_model_calls += decision.model_calls
        planner_latency += decision.latency_ms
        selected_id = decision.selected.skill_id if decision.selected else "done"
        skill = task.skill_map.get(selected_id)
        valid = bool(skill and skill.applicable(state) and skill.useful(state))
        event: dict[str, Any] = {
            "action": action_index,
            "kind": "nominal",
            "selected": selected_id,
            "route": decision.route,
            "valid": valid,
            "state_before": sorted(state_before),
        }
        if not valid or skill is None:
            nominal_memory.mark_failure(selected_id)
            event["outcome"] = "invalid"
            trace.append(event)
            continue

        attempts[selected_id] = attempts.get(selected_id, 0) + 1
        should_inject = False
        injection_reason = ""
        if selected_id == failure.skill_id:
            if forced_failure_pending:
                should_inject = True
                forced_failure_pending = False
                injection_reason = "guaranteed_protocol_fault"
            elif rng.random() < disturbance_rate:
                should_inject = True
                injection_reason = "repeated_disturbance"

            if active_failure:
                handoff_checks += 1
                contract_fraction = scenario.contract_fraction(state)
                contract_satisfied += contract_fraction
                boundary_risk = scenario.handoff_failure_probability * (1.0 - contract_fraction)
                if rng.random() < boundary_risk:
                    should_inject = True
                    injection_reason = "handoff_contract_violation"
                    handoff_failures += 1

        if should_inject:
            state = _inject(state, failure)
            state = frozenset(
                fact
                for fact in state
                if fact not in scenario.handoff_requires and not fact.startswith("recovery:")
            )
            nominal_memory.mark_failure(selected_id)
            if not active_failure:
                failures += 1
                active_started = action_index
            active_failure = True
            event.update(
                {
                    "outcome": "injected_failure",
                    "failure": failure.label,
                    "failure_reason": injection_reason,
                    "state_after": sorted(state),
                }
            )
        elif rng.random() <= skill.success_probability:
            state = skill.apply(state)
            event.update({"outcome": "executed", "state_after": sorted(state)})
        else:
            nominal_memory.mark_failure(selected_id)
            event.update({"outcome": "stochastic_skill_failure", "state_after": sorted(state)})
        trace.append(event)

    if active_failure and failure.recovered(state):
        recovered += 1
        recovery_latency_actions += len(trace) - (
            active_started if active_started is not None else len(trace)
        )

    return RecoveryEpisodeResult(
        method=method,
        task_id=task.task_id,
        seed=seed,
        disturbance_rate=disturbance_rate,
        success=task.complete(state),
        progress=task.progress(state),
        actions=actions_taken,
        failures=failures,
        recovered_failures=min(recovered, failures),
        recovery_actions=recovery_actions,
        recovery_decisions=recovery_decisions,
        recovery_latency_actions=recovery_latency_actions,
        rollback_actions=rollback_actions,
        handoff_checks=handoff_checks,
        handoff_failures=handoff_failures,
        contract_satisfied=contract_satisfied,
        monitor_queries=monitor_queries,
        recovery_model_calls=recovery_model_calls,
        nominal_model_calls=nominal_model_calls,
        planner_latency_ms=planner_latency,
        recovery_latency_ms=recovery_latency,
        memory_hits=memory_hits,
        escalations=escalations,
        trace=trace,
    )


def recovery_episode_job(
    raw_task: dict[str, Any], raw_scenario: dict[str, Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    task = TaskSpec.from_dict(raw_task)
    scenario = RecoveryScenario.from_dict(raw_scenario)
    return run_recovery_episode(task, scenario, **kwargs).to_dict()


def scenario_to_dict(scenario: RecoveryScenario) -> dict[str, Any]:
    return {
        "task_id": scenario.task_id,
        "failure_label": scenario.failure_label,
        "failed_skill_id": scenario.failed_skill_id,
        "handoff_requires": sorted(scenario.handoff_requires),
        "handoff_failure_probability": scenario.handoff_failure_probability,
        "recovery_primitives": [
            {
                "id": item.primitive_id,
                "instruction": item.instruction,
                "requires": sorted(item.requires),
                "adds": sorted(item.adds),
                "deletes": sorted(item.deletes),
                "cost": item.cost,
                "risk": item.risk,
                "success_probability": item.success_probability,
                "precompiled": item.precompiled,
            }
            for item in scenario.recovery_primitives
        ],
    }


__all__ = ["RecoveryEpisodeResult", "recovery_episode_job", "run_recovery_episode", "task_to_dict"]
