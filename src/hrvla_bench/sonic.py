"""Honest ingestion of official GEAR-SONIC motion-evaluation output."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any


def sonic_records(
    metrics_path: Path,
    *,
    checkpoint_id: str,
    controller_id: str,
    simulator_revision: str,
    run_id: str,
    recorded_at: str | None = None,
) -> list[dict[str, Any]]:
    """Map each official sample motion to a nominal benchmark episode.

    This adapter deliberately emits only the controller-tracking track. It never
    relabels motion completion as manipulation or recovery success.
    """
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    per_motion = metrics.get("eval/all_metrics_dict")
    if not isinstance(per_motion, dict):
        raise ValueError("missing eval/all_metrics_dict in SONIC metrics")
    keys = per_motion.get("motion_keys")
    terminated = per_motion.get("terminated")
    progress = per_motion.get("progress")
    local_mpjpe = per_motion.get("mpjpe_l")
    if not all(isinstance(value, list) for value in (keys, terminated, progress, local_mpjpe)):
        raise ValueError("SONIC per-motion arrays are missing")
    if len({len(keys), len(terminated), len(progress), len(local_mpjpe)}) != 1:
        raise ValueError("SONIC per-motion arrays have different lengths")

    source_metrics_sha256 = _sha256(metrics_path)
    timestamp = recorded_at or datetime.now(timezone.utc).isoformat()
    records = []
    for index, motion_key in enumerate(keys):
        success = not bool(terminated[index]) and float(progress[index]) >= 1.0
        records.append(
            {
                "schema_version": "1.0",
                "run_id": run_id,
                "episode_id": f"{run_id}-{index:04d}",
                "method_id": "gear_sonic_original_release",
                "evaluation_track": "controller_tracking_pilot",
                "suite_id": "sonic_official_sample_v0",
                "task_id": "walk_forward_motion_tracking",
                "scenario_id": str(motion_key),
                "protocol": "nominal",
                "training_seed": 0,
                "rollout_seed": index,
                "initial_snapshot_id": f"official-motion-start:{motion_key}",
                "failure_snapshot_id": None,
                "policy_checkpoint_id": checkpoint_id,
                "controller_id": controller_id,
                "simulator_revision": simulator_revision,
                "success": success,
                "termination": "success" if success else "fall",
                "failure": None,
                "completed_subtasks": int(success),
                "total_subtasks": 1,
                "diagnostics": {
                    "motion_progress": float(progress[index]),
                    "local_mpjpe_mm": float(local_mpjpe[index]),
                    "source_metrics_sha256": source_metrics_sha256,
                },
                "recorded_at": timestamp,
            }
        )
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _motion_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    values = metrics.get("eval/all_metrics_dict")
    if not isinstance(values, dict):
        raise ValueError("missing eval/all_metrics_dict in SONIC metrics")
    required = ("motion_keys", "terminated", "progress", "mpjpe_g", "mpjpe_l", "mpjpe_pa")
    if any(not isinstance(values.get(key), list) for key in required):
        raise ValueError("SONIC per-motion arrays are missing")
    lengths = {len(values[key]) for key in required}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("SONIC per-motion arrays have invalid lengths")
    episodes = len(values["motion_keys"])
    successes = sum(
        not bool(terminated) and float(progress) >= 1.0
        for terminated, progress in zip(values["terminated"], values["progress"])
    )
    return {
        "motions": list(values["motion_keys"]),
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "mean_progress": statistics.fmean(float(value) for value in values["progress"]),
        "terminations": sum(bool(value) for value in values["terminated"]),
        "mean_mpjpe_global_mm": statistics.fmean(
            float(value) for value in values["mpjpe_g"]
        ),
        "mean_mpjpe_local_mm": statistics.fmean(
            float(value) for value in values["mpjpe_l"]
        ),
        "mean_mpjpe_pa_mm": statistics.fmean(
            float(value) for value in values["mpjpe_pa"]
        ),
    }


def disturbance_pilot_report(
    nominal_metrics_path: Path,
    disturbed_metrics_path: Path,
    audit_path: Path,
    *,
    checkpoint_id: str,
    controller_id: str,
    simulator_revision: str,
    run_id: str,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Build a non-claim-bearing report for an audited SONIC body push.

    The official sample is motion tracking, not an admitted recovery task. The
    output is therefore kept outside claim-bearing episode JSONL and cannot be
    passed to the recovery scorer by accident.
    """
    nominal = _motion_summary(json.loads(nominal_metrics_path.read_text(encoding="utf-8")))
    disturbed = _motion_summary(
        json.loads(disturbed_metrics_path.read_text(encoding="utf-8"))
    )
    if nominal["motions"] != disturbed["motions"]:
        raise ValueError("nominal and disturbed runs do not contain the same motions")
    audits = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not audits or any(item.get("event") != "push_once_by_setting_velocity" for item in audits):
        raise ValueError("no valid audited push event was recorded")
    injected_envs = sum(len(item.get("environment_ids", [])) for item in audits)
    if injected_envs != disturbed["episodes"]:
        raise ValueError(
            f"audited pushes ({injected_envs}) do not match episodes ({disturbed['episodes']})"
        )
    metric_names = (
        "success_rate",
        "mean_progress",
        "terminations",
        "mean_mpjpe_global_mm",
        "mean_mpjpe_local_mm",
        "mean_mpjpe_pa_mm",
    )
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "evaluation_track": "controller_disturbance_pilot",
        "claim_status": "diagnostic_only_unadmitted",
        "method_id": "gear_sonic_original_release",
        "task_id": "walk_forward_motion_tracking",
        "protocol": "online_failure_probe",
        "failure": {
            "level": None,
            "humanoid_axes": ["H1"],
            "severity": "low",
            "event_boundary": "2.0 s after motion initialization (gait-settle timer proxy)",
            "injector_id": "isaaclab.push_by_setting_velocity+hrvla.once",
            "velocity_delta_mps": {"x": 0.0, "y": 0.35, "z": 0.0},
        },
        "injection_verified": True,
        "injection_audit": audits,
        "policy_checkpoint_id": checkpoint_id,
        "controller_id": controller_id,
        "simulator_revision": simulator_revision,
        "artifacts": {
            "nominal_metrics": {
                "path": nominal_metrics_path.as_posix(),
                "sha256": _sha256(nominal_metrics_path),
            },
            "disturbed_metrics": {
                "path": disturbed_metrics_path.as_posix(),
                "sha256": _sha256(disturbed_metrics_path),
            },
            "injection_audit": {
                "path": audit_path.as_posix(),
                "sha256": _sha256(audit_path),
            },
        },
        "nominal": nominal,
        "disturbed": disturbed,
        "delta_disturbed_minus_nominal": {
            key: disturbed[key] - nominal[key] for key in metric_names
        },
        "limitations": [
            "Two official motions are a smoke test, not a statistically powered benchmark.",
            "The timer is a proxy for gait-settle, not a task-semantic event detector.",
            "Motion completion does not establish VLA manipulation or recovery success.",
            "The disturbance is not admitted until an independent oracle passes 20/20 trials.",
        ],
        "recorded_at": recorded_at or datetime.now(timezone.utc).isoformat(),
    }
