"""Validation and result normalization for the internal HumanoidArena matrix."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .internal_protocol import INTERNAL_METHODS
from .methods import EXPECTED_FEATURES
from .plan import canonical_sha256
from .score import validate_record


TRAINING_SEEDS = (0, 1, 2)


def _is_digest(value: Any, lengths: tuple[int, ...] = (40, 64)) -> bool:
    return (
        isinstance(value, str)
        and len(value) in lengths
        and not (set(value) - set("0123456789abcdef"))
    )


def load_checkpoint_lock(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("internal checkpoint lock schema_version must be 1")
    return value


def validate_ready_checkpoint_lock(
    lock: dict[str, Any],
    *,
    method_program_sha256: str,
    internal_protocol_sha256: str,
) -> None:
    """Refuse unresolved paths, mutable revisions, or uncontrolled checkpoint reuse."""

    if lock.get("schema_version") != 1:
        raise ValueError("internal checkpoint lock schema_version must be 1")
    if lock.get("status") != "ready_for_frozen_execution":
        raise ValueError("internal checkpoints are not ready for frozen execution")
    if lock.get("method_program_sha256") != method_program_sha256:
        raise ValueError("method-program hash differs from the checkpoint lock")
    if lock.get("internal_protocol_sha256") != internal_protocol_sha256:
        raise ValueError("internal-protocol hash differs from the checkpoint lock")
    if lock.get("training_seeds") != list(TRAINING_SEEDS):
        raise ValueError("checkpoint lock training seeds must remain [0, 1, 2]")
    if not _is_digest(lock.get("simulator_revision"), (40,)):
        raise ValueError("simulator revision must be a full Git SHA")
    if not isinstance(lock.get("controller_id"), str) or "@" not in lock["controller_id"]:
        raise ValueError("controller ID must bind an immutable revision")
    device = lock.get("policy_device")
    if device not in {"cpu", "cuda:0"}:
        raise ValueError("policy device must be frozen to cpu or cuda:0")
    probe = lock.get("coexistence_probe", {})
    allowed_probe_states = {"passed"} if device == "cuda:0" else {
        "passed",
        "failed_cpu_fallback",
    }
    if probe.get("status") not in allowed_probe_states:
        raise ValueError("GR00T/Isaac coexistence probe has not resolved the policy device")
    peak = probe.get("measured_peak_compute_vram_mib")
    limit = probe.get("maximum_peak_compute_vram_mib")
    if not all(type(value) in {int, float} for value in (peak, limit)):
        raise ValueError("coexistence probe VRAM values are missing")
    if device == "cuda:0" and float(peak) > float(limit):
        raise ValueError("coexistence probe exceeds the frozen VRAM safety limit")
    if device == "cpu" and probe.get("status") == "failed_cpu_fallback" and float(peak) <= float(limit):
        gpu_attempt = probe.get("gpu_attempt", {})
        if gpu_attempt.get("status") != "fail" or not gpu_attempt.get("error"):
            raise ValueError("CPU fallback contradicts a successful probe below the safety limit")
    signatures = lock.get("method_runtime_signatures")
    if not isinstance(signatures, dict) or not signatures:
        raise ValueError("method runtime signatures are required")
    for relative, digest in signatures.items():
        if not isinstance(relative, str) or not relative or not _is_digest(digest, (64,)):
            raise ValueError("method runtime signature is malformed")

    checkpoints = lock.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != set(INTERNAL_METHODS):
        raise ValueError("checkpoint lock must contain exactly the five internal methods")
    by_method_seed: dict[tuple[str, int], dict[str, Any]] = {}
    for method_id in INTERNAL_METHODS:
        rows = checkpoints[method_id]
        if not isinstance(rows, dict) or set(rows) != {"0", "1", "2"}:
            raise ValueError(f"{method_id}: checkpoints must cover seeds 0, 1, and 2")
        for seed in TRAINING_SEEDS:
            row = rows[str(seed)]
            for key in (
                "checkpoint_id",
                "path",
                "manifest_path",
                "published_revision",
                "manifest_sha256",
            ):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError(f"{method_id}/seed-{seed}: {key} is required")
            if not _is_digest(row["published_revision"]):
                raise ValueError(f"{method_id}/seed-{seed}: published revision is mutable")
            if not _is_digest(row["manifest_sha256"], (64,)):
                raise ValueError(f"{method_id}/seed-{seed}: manifest hash is malformed")
            if row.get("training_seed") != seed:
                raise ValueError(f"{method_id}/seed-{seed}: training seed differs")
            by_method_seed[(method_id, seed)] = row

    for seed in TRAINING_SEEDS:
        identity_fields = (
            "checkpoint_id",
            "path",
            "manifest_path",
            "published_revision",
            "manifest_sha256",
        )
        common = [
            tuple(by_method_seed[(method_id, seed)][key] for key in identity_fields)
            for method_id in ("gr00t_sonic", "gr00t_st", "gr00t_str")
        ]
        if len(set(common)) != 1:
            raise ValueError(f"seed-{seed}: base, ST, and STR must share the common checkpoint")
        st_rt = by_method_seed[("gr00t_st_rt", seed)]["checkpoint_id"]
        str_rt = by_method_seed[("gr00t_str_rt", seed)]["checkpoint_id"]
        if st_rt == common[0][0] or str_rt == common[0][0] or st_rt == str_rt:
            raise ValueError(f"seed-{seed}: RT checkpoint families are not distinct")


def group_plan_cells(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Group a method-independent plan into persistent-simulator batches."""

    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for episode in plan["episodes"]:
        key = (
            episode["training_seed"],
            episode["task_id"],
            episode["scenario_id"],
            episode["protocol"],
        )
        cells[key].append(episode)
    output = []
    for key in sorted(cells):
        episodes = sorted(cells[key], key=lambda row: row["rollout_seed"])
        if len({row["rollout_seed"] for row in episodes}) != len(episodes):
            raise ValueError(f"duplicate rollout seed in internal cell {key}")
        output.append(
            {
                "training_seed": key[0],
                "task_id": key[1],
                "scenario_id": key[2],
                "protocol": key[3],
                "episodes": episodes,
            }
        )
    return output


def _termination(result: dict[str, Any]) -> str:
    if bool(result.get("success")):
        return "success"
    reason = str(result.get("failure_reason", "other")).lower()
    if "fall" in reason:
        return "fall"
    if "safety" in reason or "collision" in reason:
        return "safety"
    if "timeout" in reason or "max_step" in reason:
        return "timeout"
    if "unrecoverable" in reason:
        return "unrecoverable"
    return "other"


def normalize_internal_record(
    plan: dict[str, Any],
    episode: dict[str, Any],
    result: dict[str, Any],
    *,
    method_id: str,
    checkpoint: dict[str, Any],
    run_id: str,
    episode_id: str,
    controller_id: str,
    simulator_revision: str,
    method_program_sha256: str,
) -> dict[str, Any]:
    """Convert one audited upstream result to the shared paper record schema."""

    if method_id not in INTERNAL_METHODS:
        raise ValueError("result method is outside the frozen internal taxonomy")
    if int(result.get("episode_seed", -1)) != int(episode["rollout_seed"]):
        raise ValueError("upstream episode seed differs from the frozen rollout seed")
    method = result.get("hrvla_method")
    if not isinstance(method, dict):
        raise ValueError("upstream result lacks the internal method summary")
    expected_method = {
        "method_id": method_id,
        "task_id": episode["task_id"],
        "program_sha256": method_program_sha256,
        "policy_reset_seed": episode["rollout_seed"],
    }
    for key, expected in expected_method.items():
        if method.get(key) != expected:
            raise ValueError(f"internal method summary {key} differs")
    features = EXPECTED_FEATURES[method_id]
    if method.get("features") != features:
        raise ValueError("internal method summary feature taxonomy differs")
    if checkpoint.get("training_seed") != episode["training_seed"]:
        raise ValueError("checkpoint training seed differs from the plan")

    protocol = episode["protocol"]
    recovery = result.get("hrvla_recovery")
    if protocol == "nominal":
        if recovery is not None:
            raise ValueError("nominal internal result unexpectedly contains a recovery runtime")
        restore = result.get("hrvla_start_restore")
        if not isinstance(restore, dict) or not _is_digest(restore.get("audit_sha256"), (64,)):
            raise ValueError("nominal internal result lacks start-state restore evidence")
    else:
        if not isinstance(recovery, dict):
            raise ValueError("recovery result lacks runtime/restore evidence")
        if recovery.get("scenario_id") != episode["scenario_id"]:
            raise ValueError("recovery result scenario differs from the plan")

    success = bool(result.get("success"))
    if success != (str(result.get("failure_reason")) == "success"):
        raise ValueError("upstream success and failure reason disagree")
    total_subtasks = int(method.get("total_transition_count", 0)) + 1
    completed = min(
        total_subtasks,
        int(method.get("completed_transition_count", 0)) + int(success),
    )
    policy_latencies = [float(value) for value in method.get("policy_request_latencies_ms", [])]
    planner_latencies = [float(value) for value in method.get("planner_latencies_ms", [])]
    if (
        len(policy_latencies) != int(method.get("policy_requests", 0))
        or len(planner_latencies) != int(method.get("planner_calls", 0))
        or any(not math.isfinite(value) or value < 0.0 for value in policy_latencies)
        or any(not math.isfinite(value) or value < 0.0 for value in planner_latencies)
    ):
        raise ValueError("internal method latency evidence is incomplete or invalid")
    outcome: dict[str, Any] = {
        "success": success,
        "termination": _termination(result),
        "completed_subtasks": completed,
        "total_subtasks": total_subtasks,
        "replans": max(0, int(method.get("planner_decisions", 0)) - 1),
        "retries": int(method.get("recovery_decisions", 0)),
        "safety_violations": int(_termination(result) == "safety"),
        "wall_time_s": float(result.get("duration_sec", 0.0)),
        "policy_requests": int(method.get("policy_requests", 0)),
        "policy_request_latencies_ms": policy_latencies,
        "mean_policy_request_latency_ms": method.get("mean_policy_request_latency_ms"),
        "p95_policy_request_latency_ms": method.get("p95_policy_request_latency_ms"),
        "planner_calls": int(method.get("planner_calls", 0)),
        "planner_latencies_ms": planner_latencies,
        "mean_planner_latency_ms": method.get("mean_planner_latency_ms"),
        "p95_planner_latency_ms": method.get("p95_planner_latency_ms"),
        "diagnostics": {
            "upstream_failure_reason": str(result.get("failure_reason")),
            "episode_steps": int(result.get("episode_steps", 0)),
            "duration_sec": float(result.get("duration_sec", 0.0)),
            "final_reward": float(result.get("final_reward", 0.0)),
            "max_reward": float(result.get("max_reward", 0.0)),
            "video_recorded": bool(result.get("video_recorded")),
            "video_path": str(result.get("video_path", "")),
            "method_trace_sha256": method.get("trace_sha256"),
            "method_implementation_revision": method.get("implementation_revision"),
        },
    }
    if features["recovery"]:
        outcome["detected"] = int(method.get("recovery_decisions", 0)) > 0
        if protocol == "online_failure" and outcome["detected"]:
            trigger_step = int(recovery.get("trigger_control_step") or 0)
            decision_step = method.get("first_recovery_decision_control_step")
            if decision_step is None or int(decision_step) < trigger_step:
                raise ValueError("recovery detection step precedes the injected failure")
            outcome["detection_latency_s"] = (int(decision_step) - trigger_step) * 0.02
    if protocol != "nominal" and success and features["recovery"]:
        control_dt = 0.02
        trigger_step = 0
        if recovery.get("protocol") != "failure_start":
            trigger_step = int(recovery.get("trigger_control_step") or 0)
        elapsed_steps = max(0, int(result.get("episode_steps", 0)) - trigger_step)
        outcome["recovery_time_s"] = elapsed_steps * control_dt

    record = {
        "schema_version": "1.0",
        "run_id": run_id,
        "episode_id": episode_id,
        "method_id": method_id,
        "evaluation_track": "end_to_end_recovery",
        "suite_id": plan["suite_id"],
        "suite_sha256": plan["suite_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "task_id": episode["task_id"],
        "scenario_id": episode["scenario_id"],
        "protocol": protocol,
        "training_seed": episode["training_seed"],
        "rollout_seed": episode["rollout_seed"],
        "initial_snapshot_id": episode["initial_snapshot_id"],
        "failure_snapshot_id": episode.get("failure_snapshot_id"),
        "policy_checkpoint_id": checkpoint["checkpoint_id"],
        "controller_id": controller_id,
        "simulator_revision": simulator_revision,
        "failure": episode.get("failure"),
        **outcome,
    }
    validate_record(record)
    return record


def audit_internal_records(
    plan: dict[str, Any], method_id: str, records: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Require exactly one validated record for every planned episode of one method."""

    rows = list(records)
    expected = {
        (
            episode["task_id"],
            episode["scenario_id"],
            episode["training_seed"],
            episode["rollout_seed"],
        )
        for episode in plan["episodes"]
    }
    observed = set()
    for row in rows:
        validate_record(row)
        if row["method_id"] != method_id or row["plan_sha256"] != plan["plan_sha256"]:
            raise ValueError("internal record method or plan differs")
        key = (
            row["task_id"],
            row["scenario_id"],
            row["training_seed"],
            row["rollout_seed"],
        )
        if key in observed:
            raise ValueError("duplicate internal episode record")
        observed.add(key)
    if observed != expected:
        raise ValueError(
            f"incomplete internal records: expected={len(expected)} observed={len(observed)}"
        )
    record_hashes = sorted(canonical_sha256(row) for row in rows)
    core = {
        "schema_version": 1,
        "method_id": method_id,
        "plan_sha256": plan["plan_sha256"],
        "episode_records": len(rows),
        "record_sha256": record_hashes,
        "records_sha256": canonical_sha256(record_hashes),
        "complete": True,
    }
    return {**core, "audit_sha256": canonical_sha256(core)}


__all__ = [
    "TRAINING_SEEDS",
    "audit_internal_records",
    "group_plan_cells",
    "load_checkpoint_lock",
    "normalize_internal_record",
    "validate_ready_checkpoint_lock",
]
