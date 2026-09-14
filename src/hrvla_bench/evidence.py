"""Claim-readiness and paired-evidence gates for HRVLA evaluations."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from .execution import validate_plan_hash
from .plan import canonical_sha256, validate_suite
from .score import validate_record


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def claim_readiness(suite: dict[str, Any]) -> dict[str, Any]:
    """Return a machine-readable list of blockers to claim-bearing execution."""
    validate_suite(suite)
    blockers: list[str] = []
    admitted_tasks = 0
    admitted_scenarios = 0

    for task in suite["tasks"]:
        task_id = task["id"]
        admission = task["admission"]
        if admission["status"] == "rejected":
            continue
        if admission["status"] != "admitted":
            blockers.append(f"task {task_id}: status is {admission['status']!r}")
            continue
        admitted_tasks += 1
        if str(task["initial_snapshot_id"]).startswith("pending:"):
            blockers.append(f"task {task_id}: initial snapshot is pending")
        if not _is_sha256(admission.get("initial_snapshot_sha256")):
            blockers.append(f"task {task_id}: initial_snapshot_sha256 is required")
        if not admission.get("predicate_test_id"):
            blockers.append(f"task {task_id}: predicate_test_id is required")

        for scenario in task.get("scenarios", []):
            scenario_id = scenario["id"]
            scenario_admission = scenario["admission"]
            if scenario_admission["status"] == "rejected":
                continue
            if scenario_admission["status"] != "admitted":
                blockers.append(
                    f"scenario {scenario_id}: status is {scenario_admission['status']!r}"
                )
                continue
            admitted_scenarios += 1
            trials = scenario_admission.get("oracle_trials")
            successes = scenario_admission.get("oracle_successes")
            if type(trials) is not int or trials < 20:
                blockers.append(f"scenario {scenario_id}: at least 20 oracle trials required")
            if successes != trials:
                blockers.append(f"scenario {scenario_id}: oracle must pass every trial")
            if not scenario_admission.get("injector_revision"):
                blockers.append(f"scenario {scenario_id}: injector_revision is required")
            if not scenario_admission.get("predicate_test_id"):
                blockers.append(f"scenario {scenario_id}: predicate_test_id is required")
            if scenario["protocol"] == "failure_start":
                if str(scenario.get("failure_snapshot_id", "")).startswith("pending:"):
                    blockers.append(f"scenario {scenario_id}: failure snapshot is pending")
                if not _is_sha256(scenario_admission.get("failure_snapshot_sha256")):
                    blockers.append(
                        f"scenario {scenario_id}: failure_snapshot_sha256 is required"
                    )

    if admitted_tasks == 0:
        blockers.append("suite has no admitted tasks")
    if admitted_scenarios == 0:
        blockers.append("suite has no admitted recovery scenarios")

    core = {
        "schema_version": "1.0",
        "suite_id": suite["suite_id"],
        "suite_sha256": canonical_sha256(suite),
        "claim_ready": not blockers,
        "admitted_tasks": admitted_tasks,
        "admitted_scenarios": admitted_scenarios,
        "blockers": blockers,
    }
    return {**core, "audit_sha256": canonical_sha256(core)}


def _episode_key(value: dict[str, Any]) -> tuple[Any, ...]:
    return (
        value["task_id"],
        value["scenario_id"],
        value["protocol"],
        value["training_seed"],
        value["rollout_seed"],
        value["initial_snapshot_id"],
        value.get("failure_snapshot_id"),
    )


def audit_evidence(
    plan: dict[str, Any], records: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Prove that every planned method has one matched, claim-eligible record."""
    validate_plan_hash(plan)
    if len(plan.get("methods", [])) < 2:
        raise ValueError("claim-bearing comparison requires at least two methods")
    if not plan.get("episodes"):
        raise ValueError("claim-bearing plan contains no episodes")
    drafts = [
        episode["scenario_id"]
        for episode in plan["episodes"]
        if episode.get("admission_status") != "admitted"
    ]
    if drafts:
        scenarios = sorted(set(drafts))
        raise ValueError(f"claim-bearing plan contains non-admitted scenarios: {scenarios}")

    expected_episodes = {_episode_key(episode): episode for episode in plan["episodes"]}
    if len(expected_episodes) != len(plan["episodes"]):
        raise ValueError("plan contains duplicate episode keys")
    expected = {(method, key) for method in plan["methods"] for key in expected_episodes}

    materialized = list(records)
    if not materialized:
        raise ValueError("no episode records supplied")
    observed: Counter[tuple[str, tuple[Any, ...]]] = Counter()
    controller_simulator: set[tuple[str, str]] = set()
    checkpoints: dict[tuple[str, int], set[str]] = defaultdict(set)
    for record in materialized:
        validate_record(record)
        if record["evaluation_track"] != "end_to_end_recovery":
            raise ValueError("claim audit accepts only end_to_end_recovery records")
        if record["suite_id"] != plan["suite_id"]:
            raise ValueError(f"record suite_id does not match plan: {record['suite_id']!r}")
        if record.get("suite_sha256") != plan["suite_sha256"]:
            raise ValueError("record suite_sha256 does not match plan")
        if record.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("record plan_sha256 does not match plan")
        if record["method_id"] not in plan["methods"]:
            raise ValueError(f"unplanned method in evidence: {record['method_id']!r}")
        episode_key = _episode_key(record)
        key = (record["method_id"], episode_key)
        if key not in expected:
            raise ValueError(f"unplanned episode in evidence: {key!r}")
        planned = expected_episodes[episode_key]
        if record.get("failure") != planned.get("failure"):
            raise ValueError(
                f"failure contract does not match plan for {record['method_id']!r}/"
                f"{record['scenario_id']!r}"
            )
        observed[key] += 1
        controller_simulator.add((record["controller_id"], record["simulator_revision"]))
        checkpoints[(record["method_id"], record["training_seed"])].add(
            record["policy_checkpoint_id"]
        )

    duplicates = [key for key, count in observed.items() if count != 1]
    if duplicates:
        raise ValueError(f"duplicate method/episode evidence: {duplicates[0]!r}")
    missing = expected - set(observed)
    if missing:
        by_method = Counter(method for method, _ in missing)
        raise ValueError(f"incomplete paired evidence: missing records by method {dict(by_method)}")
    if len(controller_simulator) != 1:
        raise ValueError("controller or simulator differs across paired methods")
    inconsistent = [key for key, values in checkpoints.items() if len(values) != 1]
    if inconsistent:
        raise ValueError(
            "policy checkpoint changed within a method/training-seed cell: "
            f"{inconsistent[0]!r}"
        )

    controller_id, simulator_revision = next(iter(controller_simulator))
    core = {
        "schema_version": "1.0",
        "claim_eligible": True,
        "suite_id": plan["suite_id"],
        "suite_sha256": plan["suite_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "methods": sorted(plan["methods"]),
        "episodes_per_method": len(expected_episodes),
        "episode_records": len(materialized),
        "controller_id": controller_id,
        "simulator_revision": simulator_revision,
    }
    return {**core, "audit_sha256": canonical_sha256(core)}
