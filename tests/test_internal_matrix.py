from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hrvla_bench.hidden_final_gate import (
    build_hidden_final_gate,
    validate_hidden_final_gate,
)
from hrvla_bench.internal_matrix import (
    audit_internal_records,
    group_plan_cells,
    normalize_internal_record,
    validate_ready_checkpoint_lock,
)
from hrvla_bench.plan import build_plan, canonical_sha256


PROGRAM_SHA = "a" * 64
PROTOCOL_SHA = "b" * 64


def _admitted_suite() -> dict:
    """Keep this unit test independent of an unrelated installed ``tests`` package."""

    return {
        "schema_version": "1.0",
        "suite_id": "internal-matrix-test",
        "controller_contract": {},
        "replication": {"training_seeds": [0], "rollouts_per_seed_per_cell": 2},
        "tasks": [
            {
                "id": "pick",
                "initial_snapshot_id": "snapshot:initial",
                "success_predicate": "object placed",
                "horizon_s": 30,
                "admission": {"status": "admitted"},
                "scenarios": [
                    {
                        "id": "drop",
                        "protocol": "failure_start",
                        "failure_snapshot_id": "snapshot:drop",
                        "event_id": "after-grasp",
                        "event_boundary": "stable grasp",
                        "level": "L3",
                        "humanoid_axes": ["H3"],
                        "severity": "medium",
                        "injector": {"id": "drop-object", "parameters": {}},
                        "admission": {
                            "status": "admitted",
                            "oracle_id": "oracle@immutable",
                            "success_rate": 1.0,
                        },
                    }
                ],
            }
        ],
    }


def _checkpoint_lock() -> dict:
    common = {seed: f"common-{seed}" for seed in range(3)}
    checkpoints = {}
    for method in (
        "gr00t_sonic",
        "gr00t_st",
        "gr00t_st_rt",
        "gr00t_str",
        "gr00t_str_rt",
    ):
        checkpoints[method] = {}
        for seed in range(3):
            checkpoint_id = (
                common[seed]
                if method in {"gr00t_sonic", "gr00t_st", "gr00t_str"}
                else f"{method}-{seed}"
            )
            checkpoints[method][str(seed)] = {
                "training_seed": seed,
                "checkpoint_id": checkpoint_id,
                "path": f"checkpoints/{checkpoint_id}",
                "manifest_path": f"manifests/{checkpoint_id}.json",
                "published_revision": "c" * 40,
                "manifest_sha256": "d" * 64,
            }
    return {
        "schema_version": 1,
        "status": "ready_for_frozen_execution",
        "method_program_sha256": PROGRAM_SHA,
        "internal_protocol_sha256": PROTOCOL_SHA,
        "training_seeds": [0, 1, 2],
        "controller_id": "sonic@revision",
        "simulator_revision": "e" * 40,
        "method_runtime_signatures": {
            "src/hrvla_bench/humanoidarena_method_runtime.py": "9" * 64,
        },
        "policy_device": "cuda:0",
        "coexistence_probe": {
            "status": "passed",
            "measured_peak_compute_vram_mib": 14000,
            "maximum_peak_compute_vram_mib": 15500,
        },
        "checkpoints": checkpoints,
    }


def _result(episode: dict, method_id: str) -> dict:
    features = {
        "gr00t_sonic": {"subtask": False, "recovery": False, "retrained": False},
        "gr00t_st": {"subtask": True, "recovery": False, "retrained": False},
        "gr00t_st_rt": {"subtask": True, "recovery": False, "retrained": True},
        "gr00t_str": {"subtask": True, "recovery": True, "retrained": False},
        "gr00t_str_rt": {"subtask": True, "recovery": True, "retrained": True},
    }[method_id]
    result = {
        "episode_seed": episode["rollout_seed"],
        "success": True,
        "failure_reason": "success",
        "episode_steps": 100,
        "duration_sec": 10.0,
        "final_reward": 1.0,
        "max_reward": 1.0,
        "video_recorded": True,
        "video_path": "evidence.mp4",
        "hrvla_method": {
            "method_id": method_id,
            "task_id": episode["task_id"],
            "program_sha256": PROGRAM_SHA,
            "policy_reset_seed": episode["rollout_seed"],
            "features": features,
            "total_transition_count": 1,
            "completed_transition_count": 1,
            "planner_decisions": 2,
            "recovery_decisions": int(features["recovery"]),
            "policy_requests": 2,
            "policy_request_latencies_ms": [10.0, 12.0],
            "planner_calls": int(features["subtask"]),
            "planner_latencies_ms": [0.5] if features["subtask"] else [],
            "trace_sha256": "f" * 64,
            "implementation_revision": "1" * 40,
        },
    }
    if episode["protocol"] == "nominal":
        result["hrvla_start_restore"] = {
            "snapshot_sha256": "2" * 64,
            "audit_sha256": "3" * 64,
        }
    else:
        result["hrvla_recovery"] = {
            "scenario_id": episode["scenario_id"],
            "protocol": episode["protocol"],
            "trigger_control_step": 20,
        }
    return result


def test_ready_checkpoint_lock_enforces_controlled_checkpoint_families() -> None:
    lock = _checkpoint_lock()
    validate_ready_checkpoint_lock(
        lock,
        method_program_sha256=PROGRAM_SHA,
        internal_protocol_sha256=PROTOCOL_SHA,
    )
    changed = copy.deepcopy(lock)
    changed["checkpoints"]["gr00t_st"]["0"]["checkpoint_id"] = "different"
    with pytest.raises(ValueError, match="share the common checkpoint"):
        validate_ready_checkpoint_lock(
            changed,
            method_program_sha256=PROGRAM_SHA,
            internal_protocol_sha256=PROTOCOL_SHA,
        )


def test_checkpoint_lock_accepts_an_audited_cpu_fallback_only_after_failed_probe() -> None:
    lock = _checkpoint_lock()
    lock["policy_device"] = "cpu"
    lock["coexistence_probe"].update(
        status="failed_cpu_fallback",
        measured_peak_compute_vram_mib=16000,
    )
    validate_ready_checkpoint_lock(
        lock,
        method_program_sha256=PROGRAM_SHA,
        internal_protocol_sha256=PROTOCOL_SHA,
    )
    lock["coexistence_probe"]["measured_peak_compute_vram_mib"] = 14000
    with pytest.raises(ValueError, match="contradicts"):
        validate_ready_checkpoint_lock(
            lock,
            method_program_sha256=PROGRAM_SHA,
            internal_protocol_sha256=PROTOCOL_SHA,
        )
    lock["coexistence_probe"]["gpu_attempt"] = {
        "status": "fail",
        "error": "RuntimeError: CUDA out of memory",
    }
    validate_ready_checkpoint_lock(
        lock,
        method_program_sha256=PROGRAM_SHA,
        internal_protocol_sha256=PROTOCOL_SHA,
    )


def test_plan_cells_group_rollouts_without_losing_pairing() -> None:
    suite = _admitted_suite()
    suite["replication"]["training_seeds"] = [0, 1, 2]
    plan = build_plan(suite, ["gr00t_str"], rollouts_override=2)
    cells = group_plan_cells(plan)
    assert len(cells) == 6
    assert all(len(cell["episodes"]) == 2 for cell in cells)


def test_normalized_records_pass_the_common_schema_and_complete_audit() -> None:
    suite = _admitted_suite()
    plan = build_plan(suite, ["gr00t_str"], rollouts_override=2)
    checkpoint = {
        "training_seed": 0,
        "checkpoint_id": "common-0",
    }
    records = []
    for index, episode in enumerate(plan["episodes"]):
        records.append(
            normalize_internal_record(
                plan,
                episode,
                _result(episode, "gr00t_str"),
                method_id="gr00t_str",
                checkpoint=checkpoint,
                run_id="run",
                episode_id=f"episode-{index}",
                controller_id="sonic@revision",
                simulator_revision="e" * 40,
                method_program_sha256=PROGRAM_SHA,
            )
        )
    audit = audit_internal_records(plan, "gr00t_str", records)
    assert audit["complete"] is True
    assert audit["episode_records"] == len(plan["episodes"])
    assert len(audit["record_sha256"]) == len(plan["episodes"])


def test_hidden_final_gate_binds_complete_validation_records(tmp_path: Path) -> None:
    protocol = json.loads(
        (Path(__file__).resolve().parents[1] / "config/humanoidarena-internal-protocol.lock.json").read_text()
    )
    methods = ["gr00t_sonic", "gr00t_st", "gr00t_st_rt", "gr00t_str", "gr00t_str_rt"]
    plan = build_plan(
        _admitted_suite(),
        methods,
        rollout_seeds=protocol["splits"]["validation"]["rollout_seeds"],
    )
    method_programs = {"schema_version": 1, "tasks": []}
    program_sha = canonical_sha256(method_programs)
    checkpoint_lock = _checkpoint_lock()
    checkpoint_lock["method_program_sha256"] = program_sha
    checkpoint_lock["internal_protocol_sha256"] = canonical_sha256(protocol)
    for method_id in methods:
        records = []
        for index, episode in enumerate(plan["episodes"]):
            result = _result(episode, method_id)
            result["hrvla_method"]["program_sha256"] = program_sha
            records.append(
                normalize_internal_record(
                    plan,
                    episode,
                    result,
                    method_id=method_id,
                    checkpoint={"training_seed": 0, "checkpoint_id": f"{method_id}-0"},
                    run_id="validation",
                    episode_id=f"{method_id}-{index}",
                    controller_id="sonic@revision",
                    simulator_revision="e" * 40,
                    method_program_sha256=program_sha,
                )
            )
        method_root = tmp_path / method_id
        method_root.mkdir()
        (method_root / "records.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
        )
        (method_root / "audit.json").write_text(
            json.dumps(audit_internal_records(plan, method_id, records))
        )
    gate = build_hidden_final_gate(
        validation_plan=plan,
        protocol=protocol,
        checkpoint_lock=checkpoint_lock,
        method_programs=method_programs,
        validation_root=tmp_path,
        source_revision_before_freeze="f" * 40,
    )
    validate_hidden_final_gate(
        gate,
        validation_plan=plan,
        protocol=protocol,
        checkpoint_lock=checkpoint_lock,
        method_programs=method_programs,
        validation_root=tmp_path,
    )
    records_path = tmp_path / methods[0] / "records.jsonl"
    records_path.write_text(records_path.read_text() + "{}\n")
    with pytest.raises((KeyError, ValueError)):
        validate_hidden_final_gate(
            gate,
            validation_plan=plan,
            protocol=protocol,
            checkpoint_lock=checkpoint_lock,
            method_programs=method_programs,
            validation_root=tmp_path,
        )
