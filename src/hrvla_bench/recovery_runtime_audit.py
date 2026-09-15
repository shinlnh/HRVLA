"""Fail-closed audit for one live HumanoidArena recovery runtime trace."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .isaac_snapshot import load_snapshot
from .plan import canonical_sha256
from .recovery_injector_contract import validate_recovery_injector_contract
from .recovery_restore import validate_restore_audit


EXPECTED_SCENE_AUDIT_EVENT = {
    "drop-object-and-root-velocity": "apply_root_local_lateral_velocity_once",
    "shift-door-assembly-handle-frame": "apply_asset_local_translation_once",
    "place-doorway-obstacle": "place_asset_relative_once",
    "root-lateral-velocity-delta": "apply_root_local_lateral_velocity_once",
    "support-foot-lateral-impulse": "apply_body_impulse_once",
    "upper-body-contact-impulse": "apply_body_impulse_once",
    "place-path-obstacle": "place_asset_relative_once",
}

INFRASTRUCTURE_FAILURES = {"interrupted", "process_error", "sim_error", "sim_stopped", "unknown"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _jsonl(path: Path, *, required: bool = True) -> list[dict[str, Any]]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def _one(rows: list[dict[str, Any]], event: str) -> dict[str, Any]:
    matches = [row for row in rows if row.get("event") == event]
    if len(matches) != 1:
        raise ValueError(f"runtime trace requires exactly one {event!r}, got {len(matches)}")
    return matches[0]


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{label} must be finite")
    return output


def _close(actual: Any, expected: float, label: str, *, tolerance: float = 1e-6) -> None:
    if not math.isclose(_finite(actual, label), float(expected), abs_tol=tolerance):
        raise ValueError(f"{label} differs from the locked suite value")


def _vector(row: dict[str, Any], key: str, width: int = 3) -> list[float]:
    value = row.get(key)
    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"injector audit {key} must be a width-{width} list")
    return [_finite(item, f"injector audit {key}") for item in value]


def _validate_scene_audit(
    row: dict[str, Any],
    scenario: dict[str, Any],
    reset: dict[str, Any],
    boundary: dict[str, Any],
) -> None:
    injector = scenario["injector"]
    injector_id = injector["id"]
    parameters = injector["parameters"]
    env_id = int(reset["environment_index"])
    if row.get("environment_ids") != [env_id]:
        raise ValueError("scene injector audit must affect exactly the traced environment")
    steps = row.get("episode_steps")
    boundary_step = int(boundary["control_step"])
    if not isinstance(steps, list) or len(steps) != 1:
        raise ValueError("scene injector audit must contain one episode step")
    if int(steps[0]) not in {boundary_step - 1, boundary_step}:
        raise ValueError("scene injector audit step is not aligned with the detector edge")

    if injector_id in {"drop-object-and-root-velocity", "root-lateral-velocity-delta"}:
        _close(row.get("lateral_mps"), parameters["lateral_mps"], "lateral_mps")
        if row.get("lateral_direction_robot") != parameters["lateral_direction_robot"]:
            raise ValueError("root lateral direction differs from the locked suite value")
        world = _vector(row, "world_velocity_delta")
        _close(math.sqrt(sum(value * value for value in world)), parameters["lateral_mps"], "world velocity delta norm")
        _close(world[2], 0.0, "world velocity delta Z")
    elif injector_id == "shift-door-assembly-handle-frame":
        if row.get("asset_name") != parameters["door_asset_name"]:
            raise ValueError("door translation targeted the wrong asset")
        distance = float(parameters["translation_m"])
        expected = [distance, 0.0, 0.0] if parameters["translation_axis_door_local"] == "x" else [0.0, distance, 0.0]
        actual = _vector(row, "local_translation_m")
        for index, value in enumerate(expected):
            _close(actual[index], value, f"door translation component {index}")
    elif injector_id == "place-doorway-obstacle":
        if row.get("asset_name") != "hrvla_doorway_obstacle" or row.get("reference_asset_name") != "door":
            raise ValueError("doorway obstacle used an unexpected asset binding")
        actual = _vector(row, "reference_local_position_m")
        for index, value in enumerate(parameters["door_local_center_m"]):
            _close(actual[index], value, f"doorway obstacle position component {index}")
        if row.get("align_orientation") is not True or row.get("preserve_height") is not False:
            raise ValueError("doorway obstacle placement flags differ from the locked contract")
    elif injector_id in {"support-foot-lateral-impulse", "upper-body-contact-impulse"}:
        if row.get("asset_name") != "robot":
            raise ValueError("body impulse targeted a non-robot asset")
        if injector_id == "support-foot-lateral-impulse":
            if row.get("body_name") not in {"left_ankle_roll_link", "right_ankle_roll_link"}:
                raise ValueError("support-foot impulse did not target an ankle-roll body")
        elif row.get("body_name") != parameters["body_name"]:
            raise ValueError("contact impulse targeted the wrong body")
        impulse = _vector(row, "impulse_world_ns")
        _close(math.sqrt(sum(value * value for value in impulse)), parameters["impulse_ns"], "impulse norm")
        _close(row.get("control_dt_s"), reset["control_dt_s"], "impulse control_dt_s")
    elif injector_id == "place-path-obstacle":
        if row.get("asset_name") != parameters["asset_name"] or row.get("reference_asset_name") != "robot":
            raise ValueError("path obstacle used an unexpected asset binding")
        expected = [float(parameters["distance_ahead_m"]), float(parameters["clearance_m"]), 0.0]
        actual = _vector(row, "reference_local_position_m")
        for index, value in enumerate(expected):
            _close(actual[index], value, f"path obstacle position component {index}")
        if row.get("align_orientation") is not True or row.get("preserve_height") is not True:
            raise ValueError("path obstacle placement flags differ from the locked contract")


def _validate_action_audit(
    rows: list[dict[str, Any]],
    scenario: dict[str, Any],
    summary: dict[str, Any],
    boundary: dict[str, Any],
) -> None:
    injector_id = scenario["injector"]["id"]
    expected_action_id = {
        "release-grasp-contact": "release-grasp-contact",
        "drop-object-and-root-velocity": "release-grasp-contact",
        "attenuate-sonic-latent": "attenuate-sonic-latent",
    }.get(injector_id)
    action_rows = [row for row in rows if row.get("event") == "action_window_applied"]
    if expected_action_id is None:
        if action_rows:
            raise ValueError("scene-only injector unexpectedly wrote action-window evidence")
        return
    if not action_rows:
        raise ValueError("action-seam injector lacks action-window trace rows")
    duration_s = float(
        scenario["injector"]["parameters"].get(
            "duration_s", scenario["injector"]["parameters"].get("release_duration_s")
        )
    )
    expected_width = 64 if expected_action_id == "attenuate-sonic-latent" else 40
    changed_count = 0
    for row in action_rows:
        if row.get("scenario_id") != scenario["id"] or row.get("episode_seed") != summary["episode_seed"]:
            raise ValueError("action-window row provenance differs from the runtime summary")
        if row.get("injector_id") != expected_action_id:
            raise ValueError("action-window row used the wrong seam injector")
        if row.get("shape") not in ([expected_width], [1, expected_width]):
            raise ValueError("action-window row has the wrong interface width")
        step = int(row.get("control_step", -1))
        if step < int(boundary["control_step"]):
            raise ValueError("action-window row precedes the semantic trigger")
        elapsed_s = _finite(row.get("elapsed_s"), "action elapsed_s")
        if elapsed_s < 0.0 or elapsed_s >= duration_s:
            raise ValueError("action-window row falls outside the locked duration")
        changed = row.get("changed")
        if not isinstance(changed, bool):
            raise ValueError("action-window changed flag must be boolean")
        input_hash = row.get("input_sha256")
        output_hash = row.get("output_sha256")
        if not all(isinstance(value, str) and len(value) == 64 for value in (input_hash, output_hash)):
            raise ValueError("action-window row has malformed action hashes")
        delta = _finite(row.get("delta_l2"), "action delta_l2")
        if changed != (input_hash != output_hash) or changed != (delta > 0.0):
            raise ValueError("action-window change flag, hashes, and delta disagree")
        changed_count += int(changed)
    if changed_count != int(summary.get("action_samples_modified", -1)):
        raise ValueError("action-window changed-row count differs from the runtime summary")
    if changed_count < 1:
        raise ValueError("action-seam injector did not numerically change any sample")


def _validate_snapshot_event(
    output_dir: Path,
    row: dict[str, Any],
    expected_sha256: str,
    *,
    task_id: str,
    event_id: str,
    episode_seed: int,
    simulator_revision: str,
    environment_index: int,
) -> tuple[str, str]:
    path = Path(str(row["snapshot_path"]))
    if not path.is_absolute():
        path = output_dir / path
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(output_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"snapshot escapes recovery output directory: {resolved}") from exc
    snapshot = load_snapshot(resolved)
    if snapshot["snapshot_sha256"] != expected_sha256:
        raise ValueError("snapshot content hash differs from the runtime summary")
    if row.get("snapshot_sha256") != expected_sha256:
        raise ValueError("snapshot trace hash differs from the runtime summary")
    expected_metadata = {
        "task_id": task_id,
        "event_id": event_id,
        "episode_seed": episode_seed,
        "simulator_revision": simulator_revision,
        "environment_index": environment_index,
    }
    for key, expected in expected_metadata.items():
        if snapshot.get(key) != expected:
            raise ValueError(f"snapshot {key} differs from the traced runtime")
    return str(relative), _sha256(resolved)


def audit_recovery_runtime_trace(
    suite: dict[str, Any],
    scenario_id: str,
    output_dir: Path,
    result_json: Path,
) -> dict[str, Any]:
    """Validate a complete trace without claiming scenario recoverability."""

    output_dir = output_dir.resolve()
    contract = next(
        (
            row
            for row in validate_recovery_injector_contract(suite)
            if row["scenario_id"] == scenario_id
        ),
        None,
    )
    if contract is None:
        raise ValueError(f"unknown recovery scenario: {scenario_id}")
    task = next(task for task in suite["tasks"] if task["id"] == contract["task_id"])
    scenario = next(item for item in task["scenarios"] if item["id"] == scenario_id)
    summary_path = output_dir / "runtime-summary.json"
    trace_path = output_dir / "runtime-trace.jsonl"
    audit_path = output_dir / "injector-audit.jsonl"
    summary = _json(summary_path)
    trace = _jsonl(trace_path)
    injector_rows = _jsonl(audit_path, required=False)
    result = _json(result_json.resolve())

    suite_sha256 = canonical_sha256(suite)
    expected_summary = {
        "suite_sha256": suite_sha256,
        "task_id": task["id"],
        "scenario_id": scenario_id,
        "detector_id": contract["detector_id"],
        "injector_id": contract["injector_id"],
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"runtime summary {key} differs: {summary.get(key)!r} != {expected!r}")
    if summary.get("triggered") is not True or summary.get("trigger_control_step") is None:
        raise ValueError("runtime summary does not prove a detector trigger")
    if summary.get("runtime_validated") is not False:
        raise ValueError("runtime summary must not self-validate")
    if result.get("hrvla_recovery") != summary:
        raise ValueError("episode result does not embed the exact runtime summary")
    if result.get("failure_reason") in INFRASTRUCTURE_FAILURES:
        raise ValueError(f"episode terminated with an infrastructure error: {result['failure_reason']}")
    if int(result.get("episode_seed", -1)) != int(summary["episode_seed"]):
        raise ValueError("episode result seed differs from the runtime summary")
    if bool(result.get("success")) != (result.get("failure_reason") == "success"):
        raise ValueError("episode result success and failure_reason disagree")

    reset = _one(trace, "episode_reset")
    boundary = _one(trace, "semantic_boundary_triggered")
    episode_seed = int(summary["episode_seed"])
    if int(reset.get("episode_seed", -1)) != episode_seed:
        raise ValueError("episode reset seed differs from runtime summary")
    if reset.get("suite_sha256") != suite_sha256:
        raise ValueError("episode reset suite hash differs")
    if int(reset.get("environment_index", -1)) != 0:
        raise ValueError("recovery admission requires the locked single environment index 0")
    if not isinstance(reset.get("simulator_revision"), str) or not reset["simulator_revision"]:
        raise ValueError("episode reset lacks simulator revision provenance")
    implementation_revision = summary.get("implementation_revision")
    if (
        not isinstance(implementation_revision, str)
        or len(implementation_revision) != 40
        or set(implementation_revision) - set("0123456789abcdef")
    ):
        raise ValueError("runtime summary lacks a full Git implementation revision")
    if reset.get("implementation_revision") != implementation_revision:
        raise ValueError("episode reset implementation revision differs from runtime summary")
    if boundary.get("scenario_id") != scenario_id:
        raise ValueError("semantic boundary scenario differs")
    if boundary.get("detector_id") != contract["detector_id"]:
        raise ValueError("semantic boundary detector differs")
    if boundary.get("injector_id") != contract["injector_id"]:
        raise ValueError("semantic boundary injector differs")
    if int(boundary.get("control_step", -1)) != int(summary["trigger_control_step"]):
        raise ValueError("semantic boundary control step differs")

    restore_report = None
    start_snapshot_sha = summary.get("start_snapshot_sha256")
    restore_audit_sha = summary.get("restore_audit_sha256")
    if (start_snapshot_sha is None) != (restore_audit_sha is None):
        raise ValueError("runtime summary has incomplete start-snapshot provenance")
    if reset.get("start_snapshot_sha256") != start_snapshot_sha:
        raise ValueError("episode reset start snapshot differs from runtime summary")
    if reset.get("restore_audit_sha256") != restore_audit_sha:
        raise ValueError("episode reset restore audit differs from runtime summary")
    restore_path = output_dir / "restore-audit.json"
    if start_snapshot_sha is not None:
        restore_report = _json(restore_path)
        claimed_restore_hash = validate_restore_audit(
            restore_report,
            expected_snapshot_sha256=str(start_snapshot_sha),
            expected_task_id=task["id"],
            expected_event_id="initial",
            expected_simulator_revision=reset["simulator_revision"],
            expected_policy_rollout_seed=episode_seed,
        )
        if claimed_restore_hash != restore_audit_sha:
            raise ValueError("restore audit hash differs from runtime summary")
    elif restore_path.exists():
        raise ValueError("capture-only runtime unexpectedly contains a restore audit")

    seam = contract["interface_seam"]
    _validate_action_audit(trace, scenario, summary, boundary)
    expected_scene_event = EXPECTED_SCENE_AUDIT_EVENT.get(contract["injector_id"])
    if expected_scene_event is None:
        if injector_rows:
            raise ValueError("action-only injector unexpectedly wrote a scene audit")
    else:
        if len(injector_rows) != 1 or injector_rows[0].get("event") != expected_scene_event:
            raise ValueError(
                f"scene injector requires one {expected_scene_event!r} audit row"
            )
        audit_row = injector_rows[0]
        if audit_row.get("injector_id") != contract["injector_id"]:
            raise ValueError("scene injector audit ID differs")
        if int(audit_row.get("episode_seed", -1)) != episode_seed:
            raise ValueError("scene injector audit seed differs")
        _validate_scene_audit(audit_row, scenario, reset, boundary)

    snapshots: dict[str, Any] = {}
    initial_sha = summary.get("initial_snapshot_sha256")
    if initial_sha is not None:
        relative, file_sha = _validate_snapshot_event(
            output_dir,
            _one(trace, "initial_snapshot_captured"),
            str(initial_sha),
            task_id=task["id"],
            event_id="initial",
            episode_seed=episode_seed,
            simulator_revision=reset["simulator_revision"],
            environment_index=int(reset["environment_index"]),
        )
        snapshots["initial"] = {
            "snapshot_sha256": initial_sha,
            "path": relative,
            "file_sha256": file_sha,
        }
    if scenario["protocol"] == "failure_start":
        if summary.get("failure_capture_complete") is not True:
            raise ValueError("failure_start runtime did not complete failure snapshot capture")
        failure_sha = summary.get("failure_snapshot_sha256")
        if not isinstance(failure_sha, str):
            raise ValueError("failure_start runtime summary lacks a failure snapshot hash")
        relative, file_sha = _validate_snapshot_event(
            output_dir,
            _one(trace, "failure_snapshot_captured"),
            failure_sha,
            task_id=task["id"],
            event_id=scenario_id,
            episode_seed=episode_seed,
            simulator_revision=reset["simulator_revision"],
            environment_index=int(reset["environment_index"]),
        )
        failure_row = _one(trace, "failure_snapshot_captured")
        expected_step = int(boundary["control_step"]) + math.ceil(
            float(scenario["failure_snapshot_settle_s"])
            / _finite(reset["control_dt_s"], "control_dt_s")
        )
        if int(failure_row.get("control_step", -1)) != expected_step:
            raise ValueError("failure snapshot was not captured at the locked settle boundary")
        snapshots["failure"] = {
            "snapshot_sha256": failure_sha,
            "path": relative,
            "file_sha256": file_sha,
        }
    elif summary.get("failure_snapshot_sha256") is not None:
        raise ValueError("online_failure runtime unexpectedly captured a failure snapshot")

    files = {
        "runtime_summary": _sha256(summary_path),
        "runtime_trace": _sha256(trace_path),
        "episode_result": _sha256(result_json.resolve()),
    }
    if injector_rows:
        files["injector_audit"] = _sha256(audit_path)
    if restore_report is not None:
        files["restore_audit"] = _sha256(restore_path)
    core = {
        "schema_version": 1,
        "status": "runtime_trace_passed_oracle_admission_pending",
        "claim_boundary": "injector runtime evidence only; not recoverability or method success",
        "suite_sha256": suite_sha256,
        "task_id": task["id"],
        "scenario_id": scenario_id,
        "protocol": scenario["protocol"],
        "episode_seed": episode_seed,
        "detector_id": contract["detector_id"],
        "injector_id": contract["injector_id"],
        "interface_seam": seam,
        "implementation_revision": implementation_revision,
        "runtime_trace_validated": True,
        "action_samples_modified": int(summary.get("action_samples_modified", 0)),
        "snapshots": snapshots,
        "artifact_file_sha256": files,
    }
    return {**core, "audit_sha256": canonical_sha256(core)}


__all__ = ["audit_recovery_runtime_trace"]
