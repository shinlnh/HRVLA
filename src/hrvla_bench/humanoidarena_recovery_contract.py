"""Static provenance gates for HumanoidArena recovery scenario predicates."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from .plan import canonical_sha256, validate_suite
from .recovery_injector_contract import validate_recovery_injector_contract


EXPECTED_TASK_KEYS = {
    "boxing",
    "doubledesk",
    "football",
    "open_door",
    "pp_box",
    "sit_sofa",
    "vision_navi",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_python_symbol(source_root: Path, reference: str) -> dict[str, Any]:
    """Resolve ``relative/path.py:symbol`` without importing Isaac Sim code."""

    relative, separator, symbol = reference.partition(":")
    if not separator or not relative or not symbol:
        raise ValueError(f"invalid predicate source reference: {reference!r}")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"predicate source must stay inside the upstream tree: {relative!r}")
    path = source_root / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    if symbol not in definitions:
        raise ValueError(f"predicate symbol {symbol!r} is missing from {relative}")
    return {
        "reference": reference,
        "path": relative,
        "symbol": symbol,
        "source_sha256": _sha256(path),
    }


def audit_recovery_contract(
    suite: dict[str, Any], source_root: Path, expected_revision: str
) -> dict[str, Any]:
    """Prove source identity/predicate resolution while keeping admission at zero."""

    validate_suite(suite)
    injector_contract_rows = validate_recovery_injector_contract(suite)
    injector_contract_by_scenario = {
        row["scenario_id"]: row for row in injector_contract_rows
    }
    revision = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != expected_revision:
        raise ValueError(f"HumanoidArena revision is {revision}, expected {expected_revision}")
    actual_keys = {str(task.get("humanoidarena_task_key", "")) for task in suite["tasks"]}
    if actual_keys != EXPECTED_TASK_KEYS:
        raise ValueError(
            f"suite task mapping differs from the released seven tasks: {sorted(actual_keys)}"
        )

    tasks = []
    scenarios = []
    failure_snapshot_count = 0
    for task in suite["tasks"]:
        source = resolve_python_symbol(source_root, task["success_predicate_source"])
        tasks.append(
            {
                "task_id": task["id"],
                "humanoidarena_task_key": task["humanoidarena_task_key"],
                "predicate": task["success_predicate"],
                "predicate_source": source,
                "admission_status": task["admission"]["status"],
            }
        )
        for scenario in task.get("scenarios", []):
            if scenario["protocol"] == "failure_start":
                failure_snapshot_count += 1
            contract = injector_contract_by_scenario[scenario["id"]]
            scenarios.append(
                {
                    "task_id": task["id"],
                    "scenario_id": scenario["id"],
                    "protocol": scenario["protocol"],
                    "injector_id": scenario["injector"]["id"],
                    "injector_parameters_sha256": _canonical_hash(
                        scenario["injector"]["parameters"]
                    ),
                    "event_detector_id": contract["detector_id"],
                    "event_detector_parameters_sha256": contract[
                        "detector_parameters_sha256"
                    ],
                    "interface_seam": contract["interface_seam"],
                    "static_contract_validated": True,
                    "admission_status": scenario["admission"]["status"],
                    "runtime_validated": False,
                    "oracle_trials": 0,
                }
            )
    if len(tasks) != 7 or len(scenarios) != 9 or failure_snapshot_count != 3:
        raise ValueError("suite must retain 7 tasks, 9 scenarios, and 3 failure-start snapshots")

    return {
        "schema_version": 1,
        "status": "predicate_sources_passed_runtime_admission_pending",
        "claim_boundary": (
            "static source provenance only; no scenario is admitted and no oracle trial is claimed"
        ),
        "suite_id": suite["suite_id"],
        "suite_sha256": canonical_sha256(suite),
        "humanoidarena_revision": revision,
        "tasks": tasks,
        "scenarios": scenarios,
        "gates": {
            "task_predicate_sources": {"complete": 7, "required": 7},
            "static_injector_contracts": {"complete": 9, "required": 9},
            "initial_snapshots": {"complete": 0, "required": 7},
            "failure_snapshots": {"complete": 0, "required": 3},
            "runtime_injectors": {"complete": 0, "required": 9},
            "oracle_20_of_20": {"complete": 0, "required": 9},
        },
        "open_door_primary_aggregate_eligible": False,
        "open_door_blocker": "two locked upstream self-tests remain failing",
    }
