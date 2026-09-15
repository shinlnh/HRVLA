"""Compile live recovery capture attempts into one hash-addressed admission manifest."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .plan import canonical_sha256
from .recovery_runtime_audit import audit_recovery_runtime_trace


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def build_capture_manifest(
    suite: dict[str, Any], captures: Mapping[str, dict[str, Any]]
) -> dict[str, Any]:
    """Validate complete runtime captures without promoting oracle admission."""

    suite_sha256 = canonical_sha256(suite)
    expected = {
        scenario["id"]: (task, scenario)
        for task in suite["tasks"]
        for scenario in task["scenarios"]
    }
    if set(captures) != set(expected):
        missing = sorted(set(expected) - set(captures))
        extra = sorted(set(captures) - set(expected))
        raise ValueError(f"capture set differs from suite: missing={missing}, extra={extra}")

    initial_by_task: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    scenarios = []
    for scenario_id in expected:
        task, scenario = expected[scenario_id]
        capture = captures[scenario_id]
        audit = capture.get("audit")
        if not isinstance(audit, dict):
            raise ValueError(f"{scenario_id}: capture lacks a runtime audit")
        if audit.get("suite_sha256") != suite_sha256:
            raise ValueError(f"{scenario_id}: runtime audit suite hash differs")
        if audit.get("task_id") != task["id"] or audit.get("scenario_id") != scenario_id:
            raise ValueError(f"{scenario_id}: runtime audit identity differs")
        if audit.get("runtime_trace_validated") is not True:
            raise ValueError(f"{scenario_id}: runtime trace is not validated")
        if not _is_sha256(audit.get("audit_sha256")):
            raise ValueError(f"{scenario_id}: runtime audit hash is malformed")
        initial = audit.get("snapshots", {}).get("initial")
        if not isinstance(initial, dict):
            raise ValueError(f"{scenario_id}: initial snapshot evidence is required")
        if not all(
            _is_sha256(initial.get(key))
            for key in ("snapshot_sha256", "file_sha256")
        ):
            raise ValueError(f"{scenario_id}: initial snapshot hashes are malformed")
        initial_by_task[task["id"]].append((scenario_id, initial))
        failure = audit.get("snapshots", {}).get("failure")
        if scenario["protocol"] == "failure_start" and not isinstance(failure, dict):
            raise ValueError(f"{scenario_id}: failure snapshot evidence is required")
        if scenario["protocol"] == "online_failure" and failure is not None:
            raise ValueError(f"{scenario_id}: online failure has a failure snapshot")
        if failure is not None and not all(
            _is_sha256(failure.get(key))
            for key in ("snapshot_sha256", "file_sha256")
        ):
            raise ValueError(f"{scenario_id}: failure snapshot hashes are malformed")
        evidence_directory = capture.get("evidence_directory")
        if not isinstance(evidence_directory, str) or not evidence_directory:
            raise ValueError(f"{scenario_id}: evidence directory is required")
        scenarios.append(
            {
                "task_id": task["id"],
                "scenario_id": scenario_id,
                "protocol": scenario["protocol"],
                "runtime_audit_sha256": audit["audit_sha256"],
                "runtime_evidence_directory": evidence_directory,
                "initial_snapshot_sha256": initial["snapshot_sha256"],
                "initial_snapshot_file_sha256": initial["file_sha256"],
                "initial_snapshot_path": (
                    f"{evidence_directory}/{initial['path']}"
                ),
                "failure_snapshot_sha256": (
                    None if failure is None else failure["snapshot_sha256"]
                ),
                "failure_snapshot_file_sha256": (
                    None if failure is None else failure["file_sha256"]
                ),
                "failure_snapshot_path": (
                    None
                    if failure is None
                    else f"{evidence_directory}/{failure['path']}"
                ),
            }
        )

    initial_snapshots = []
    for task in suite["tasks"]:
        rows = initial_by_task[task["id"]]
        state_hashes = {row[1]["snapshot_sha256"] for row in rows}
        file_hashes = {row[1]["file_sha256"] for row in rows}
        if len(state_hashes) != 1 or len(file_hashes) != 1:
            raise ValueError(
                f"{task['id']}: initial snapshots differ across scenario captures"
            )
        initial_snapshots.append(
            {
                "task_id": task["id"],
                "snapshot_sha256": next(iter(state_hashes)),
                "file_sha256": next(iter(file_hashes)),
                "proved_by_scenarios": [scenario_id for scenario_id, _ in rows],
            }
        )

    core = {
        "schema_version": 1,
        "status": "runtime_and_snapshots_complete_oracle_admission_pending",
        "claim_boundary": "capture/runtime evidence only; oracle trials remain 0/20",
        "suite_id": suite["suite_id"],
        "suite_sha256": suite_sha256,
        "gates": {
            "runtime_injectors": {"complete": len(scenarios), "required": len(expected)},
            "initial_snapshots": {
                "complete": len(initial_snapshots),
                "required": len(suite["tasks"]),
            },
            "failure_snapshots": {
                "complete": sum(row["failure_snapshot_sha256"] is not None for row in scenarios),
                "required": sum(
                    scenario["protocol"] == "failure_start"
                    for _task, scenario in expected.values()
                ),
            },
            "oracle_20_of_20": {"complete": 0, "required": len(expected)},
        },
        "initial_snapshots": initial_snapshots,
        "scenarios": scenarios,
    }
    return {**core, "manifest_sha256": canonical_sha256(core)}


def discover_valid_captures(
    suite: dict[str, Any], output_root: Path
) -> dict[str, dict[str, Any]]:
    """Re-audit the newest valid attempt for each scenario; ignore failed attempts."""

    captures: dict[str, dict[str, Any]] = {}
    for task in suite["tasks"]:
        for scenario in task["scenarios"]:
            scenario_id = scenario["id"]
            scenario_dir = output_root / scenario_id
            attempts = sorted(
                path
                for path in scenario_dir.glob("attempt-*")
                if path.is_dir() and path.name[len("attempt-") :].isdigit()
            )
            for attempt in reversed(attempts):
                result = attempt / "episode.json"
                if not result.is_file():
                    continue
                try:
                    audit = audit_recovery_runtime_trace(
                        suite, scenario_id, attempt, result
                    )
                except (
                    FileNotFoundError,
                    KeyError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ):
                    continue
                captures[scenario_id] = {
                    "audit": audit,
                    "evidence_directory": str(attempt.relative_to(output_root)),
                }
                break
    return captures


__all__ = ["build_capture_manifest", "discover_valid_captures"]
