"""Audited restoration of immutable recovery snapshots into Isaac Lab."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .isaac_snapshot import (
    capture_scene_snapshot,
    load_snapshot,
    restore_scene_snapshot,
    snapshot_state_sha256,
)
from .plan import canonical_sha256


INFRASTRUCTURE_FAILURES = {
    "interrupted",
    "process_error",
    "sim_error",
    "sim_stopped",
    "unknown",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_snapshot_for_trial(
    env: Any,
    snapshot_path: Path,
    *,
    expected_snapshot_sha256: str,
    expected_task_id: str,
    expected_event_id: str,
    expected_simulator_revision: str,
    policy_rollout_seed: int,
    audit_path: Path,
) -> dict[str, Any]:
    """Restore and immediately read back all locked state before policy execution."""

    snapshot_path = snapshot_path.resolve()
    if audit_path.exists():
        raise RuntimeError(f"refusing to overwrite restore audit: {audit_path}")
    snapshot = load_snapshot(snapshot_path)
    expected = {
        "snapshot_sha256": expected_snapshot_sha256,
        "task_id": expected_task_id,
        "event_id": expected_event_id,
        "simulator_revision": expected_simulator_revision,
        "environment_index": 0,
    }
    for key, value in expected.items():
        if snapshot.get(key) != value:
            raise ValueError(f"restore snapshot {key} differs: {snapshot.get(key)!r} != {value!r}")

    source_state_sha256 = snapshot_state_sha256(snapshot)
    restore_scene_snapshot(env, snapshot)
    readback = capture_scene_snapshot(
        env,
        task_id=snapshot["task_id"],
        event_id=snapshot["event_id"],
        episode_seed=int(snapshot["episode_seed"]),
        scene_assets=tuple(snapshot["assets"]),
        simulator_revision=snapshot["simulator_revision"],
        env_id=int(snapshot["environment_index"]),
    )
    readback_state_sha256 = snapshot_state_sha256(readback)
    core = {
        "schema_version": 1,
        "event": "snapshot_restored_and_read_back",
        "task_id": expected_task_id,
        "event_id": expected_event_id,
        "policy_rollout_seed": int(policy_rollout_seed),
        "simulator_revision": expected_simulator_revision,
        "environment_index": 0,
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": expected_snapshot_sha256,
        "snapshot_file_sha256": _file_sha256(snapshot_path),
        "source_state_sha256": source_state_sha256,
        "readback_state_sha256": readback_state_sha256,
        "restore_validated": readback_state_sha256 == source_state_sha256,
        "asset_names": sorted(snapshot["assets"]),
        "claim_boundary": "pre-policy state restoration only; not task or recovery success",
    }
    report = {**core, "audit_sha256": canonical_sha256(core)}
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = audit_path.with_suffix(audit_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, audit_path)
    if not report["restore_validated"]:
        raise RuntimeError("snapshot readback state differs immediately after restore")
    return report


def validate_restore_audit(
    restore: dict[str, Any],
    *,
    expected_snapshot_sha256: str,
    expected_task_id: str,
    expected_event_id: str,
    expected_simulator_revision: str,
    expected_policy_rollout_seed: int,
) -> str:
    """Recompute a restore report's hashes and locked provenance."""

    claimed = restore.get("audit_sha256")
    core = {key: value for key, value in restore.items() if key != "audit_sha256"}
    if claimed != canonical_sha256(core):
        raise ValueError("restore audit content hash differs")
    expected = {
        "snapshot_sha256": expected_snapshot_sha256,
        "task_id": expected_task_id,
        "event_id": expected_event_id,
        "simulator_revision": expected_simulator_revision,
        "environment_index": 0,
        "policy_rollout_seed": int(expected_policy_rollout_seed),
        "restore_validated": True,
    }
    for key, value in expected.items():
        if restore.get(key) != value:
            raise ValueError(f"restore audit {key} differs")
    source_snapshot_path = Path(str(restore.get("snapshot_path", ""))).resolve()
    source_snapshot = load_snapshot(source_snapshot_path)
    if source_snapshot["snapshot_sha256"] != expected_snapshot_sha256:
        raise ValueError("restore source snapshot content address differs")
    if _file_sha256(source_snapshot_path) != restore.get("snapshot_file_sha256"):
        raise ValueError("restore source snapshot file hash differs")
    source_state_sha = snapshot_state_sha256(source_snapshot)
    if source_state_sha != restore.get("source_state_sha256"):
        raise ValueError("restore source state hash differs")
    if restore.get("readback_state_sha256") != source_state_sha:
        raise ValueError("restore readback state hash differs")
    return str(claimed)


def audit_failure_start_trial(
    suite: dict[str, Any],
    scenario_id: str,
    output_dir: Path,
    result_path: Path,
    *,
    expected_snapshot_sha256: str,
) -> dict[str, Any]:
    """Audit a restore-only failure-start episode without claiming admission."""

    scenario_task = next(
        (
            (task, scenario)
            for task in suite["tasks"]
            for scenario in task["scenarios"]
            if scenario["id"] == scenario_id
        ),
        None,
    )
    if scenario_task is None:
        raise ValueError(f"unknown recovery scenario: {scenario_id}")
    task, scenario = scenario_task
    if scenario["protocol"] != "failure_start":
        raise ValueError("restore-only trial audit requires a failure_start scenario")
    restore_path = output_dir / "restore-audit.json"
    summary_path = output_dir / "trial-summary.json"
    restore = json.loads(restore_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not all(isinstance(value, dict) for value in (restore, summary, result)):
        raise ValueError("failure-start trial sidecars must be JSON objects")

    expected_simulator = suite["reference_stack"]["isaac_lab_revision"]
    claimed_restore_hash = validate_restore_audit(
        restore,
        expected_snapshot_sha256=expected_snapshot_sha256,
        expected_task_id=task["id"],
        expected_event_id=scenario_id,
        expected_simulator_revision=expected_simulator,
        expected_policy_rollout_seed=int(summary["episode_seed"]),
    )

    expected_summary = {
        "suite_sha256": canonical_sha256(suite),
        "task_id": task["id"],
        "scenario_id": scenario_id,
        "protocol": "failure_start",
        "failure_injected": False,
        "start_snapshot_sha256": expected_snapshot_sha256,
        "restore_audit_sha256": claimed_restore_hash,
        "restore_validated": True,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"failure-start trial summary {key} differs")
    if result.get("hrvla_recovery") != summary:
        raise ValueError("failure-start episode does not embed the exact trial summary")
    if int(result.get("episode_seed", -1)) != int(summary.get("episode_seed", -2)):
        raise ValueError("failure-start episode policy seed differs")
    if int(restore.get("policy_rollout_seed", -1)) != int(summary["episode_seed"]):
        raise ValueError("failure-start restore policy seed differs")
    if result.get("failure_reason") in INFRASTRUCTURE_FAILURES:
        raise ValueError("failure-start episode ended with an infrastructure failure")
    if not isinstance(result.get("success"), bool):
        raise ValueError("failure-start episode success must be boolean")
    if result["success"] != (result.get("failure_reason") == "success"):
        raise ValueError("failure-start episode success and reason disagree")

    core = {
        "schema_version": 1,
        "status": "restore_and_behavioral_result_validated",
        "claim_boundary": "one failure-start oracle trial; not scenario admission",
        "suite_sha256": canonical_sha256(suite),
        "task_id": task["id"],
        "scenario_id": scenario_id,
        "episode_seed": int(summary["episode_seed"]),
        "snapshot_sha256": expected_snapshot_sha256,
        "restore_audit_sha256": claimed_restore_hash,
        "success": result["success"],
        "failure_reason": result["failure_reason"],
        "artifact_file_sha256": {
            "restore_audit": _file_sha256(restore_path),
            "trial_summary": _file_sha256(summary_path),
            "episode_result": _file_sha256(result_path),
        },
    }
    return {**core, "audit_sha256": canonical_sha256(core)}


__all__ = [
    "audit_failure_start_trial",
    "restore_snapshot_for_trial",
    "validate_restore_audit",
]
