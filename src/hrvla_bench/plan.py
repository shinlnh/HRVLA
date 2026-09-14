"""Deterministic experiment-plan generation for the HRVLA benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ADMISSION_STATES = {"admitted", "draft", "rejected"}
PROTOCOLS = {"nominal", "failure_start", "online_failure"}


def canonical_sha256(value: Any) -> str:
    """Hash JSON data independently of whitespace and object key order."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be a JSON object")
    return value


def validate_suite(suite: dict[str, Any]) -> None:
    """Validate invariants needed to generate fair paired episodes."""
    if suite.get("schema_version") != "1.0":
        raise ValueError("suite schema_version must be '1.0'")
    if not suite.get("suite_id"):
        raise ValueError("suite_id is required")
    replication = suite.get("replication")
    if not isinstance(replication, dict):
        raise ValueError("replication is required")
    seeds = replication.get("training_seeds")
    rollouts = replication.get("rollouts_per_seed_per_cell")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("training_seeds must be a non-empty integer list")
    if type(rollouts) is not int or rollouts < 1:
        raise ValueError("rollouts_per_seed_per_cell must be positive")

    tasks = suite.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a non-empty list")
    task_ids: set[str] = set()
    scenario_ids: set[str] = set()
    for task in tasks:
        task_id = task.get("id")
        if not task_id or task_id in task_ids:
            raise ValueError(f"task id is missing or duplicated: {task_id!r}")
        task_ids.add(task_id)
        if not task.get("initial_snapshot_id") or not task.get("success_predicate"):
            raise ValueError(f"{task_id}: initial_snapshot_id and success_predicate are required")
        if task.get("admission", {}).get("status") not in ADMISSION_STATES:
            raise ValueError(f"{task_id}: invalid task admission status")
        for scenario in task.get("scenarios", []):
            scenario_id = scenario.get("id")
            if not scenario_id or scenario_id in scenario_ids:
                raise ValueError(f"scenario id is missing or duplicated: {scenario_id!r}")
            scenario_ids.add(scenario_id)
            if scenario.get("protocol") not in PROTOCOLS - {"nominal"}:
                raise ValueError(f"{scenario_id}: recovery scenario has invalid protocol")
            admission = scenario.get("admission", {})
            if admission.get("status") not in ADMISSION_STATES:
                raise ValueError(f"{scenario_id}: invalid admission status")
            if admission.get("status") == "admitted":
                if not admission.get("oracle_id") or admission.get("success_rate") != 1.0:
                    raise ValueError(
                        f"{scenario_id}: admitted scenarios require an oracle with success_rate 1.0"
                    )


def _scenario_failure(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": scenario["event_id"],
        "event_boundary": scenario["event_boundary"],
        "level": scenario["level"],
        "humanoid_axes": scenario["humanoid_axes"],
        "severity": scenario["severity"],
        "recoverable_oracle": scenario["admission"]["status"] == "admitted",
        "injector_id": scenario["injector"]["id"],
        "injector_parameters": scenario["injector"]["parameters"],
    }


def build_plan(
    suite: dict[str, Any],
    methods: Iterable[str],
    *,
    include_draft: bool = False,
    rollouts_override: int | None = None,
) -> dict[str, Any]:
    """Expand a suite into paired, method-independent episode specifications."""
    validate_suite(suite)
    method_ids = sorted(set(methods))
    if not method_ids or any(not method for method in method_ids):
        raise ValueError("at least one non-empty method id is required")
    replication = suite["replication"]
    rollouts = rollouts_override or replication["rollouts_per_seed_per_cell"]
    if rollouts < 1:
        raise ValueError("rollouts_override must be positive")

    episodes: list[dict[str, Any]] = []
    for task in suite["tasks"]:
        scenarios = [
            {
                "id": f"{task['id']}--nominal",
                "protocol": "nominal",
                "admission": {"status": task["admission"]["status"]},
            },
            *task.get("scenarios", []),
        ]
        for scenario in scenarios:
            status = scenario["admission"]["status"]
            if status == "rejected" or (status == "draft" and not include_draft):
                continue
            for training_seed in replication["training_seeds"]:
                for rollout_seed in range(rollouts):
                    episode_key = (
                        f"{task['id']}::{scenario['id']}::t{training_seed}::r{rollout_seed}"
                    )
                    episode: dict[str, Any] = {
                        "episode_key": episode_key,
                        "task_id": task["id"],
                        "scenario_id": scenario["id"],
                        "protocol": scenario["protocol"],
                        "training_seed": training_seed,
                        "rollout_seed": rollout_seed,
                        "initial_snapshot_id": task["initial_snapshot_id"],
                        "success_predicate": task["success_predicate"],
                        "horizon_s": task["horizon_s"],
                        "admission_status": status,
                    }
                    if scenario["protocol"] != "nominal":
                        episode["failure"] = _scenario_failure(scenario)
                        if scenario["protocol"] == "failure_start":
                            episode["failure_snapshot_id"] = scenario["failure_snapshot_id"]
                    episodes.append(episode)

    plan_core = {
        "schema_version": "1.0",
        "suite_id": suite["suite_id"],
        "suite_sha256": canonical_sha256(suite),
        "methods": method_ids,
        "controller_contract": suite["controller_contract"],
        "episodes": episodes,
    }
    return {**plan_core, "plan_sha256": canonical_sha256(plan_core)}


def write_plan(plan: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
