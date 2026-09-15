from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hrvla_bench.plan import canonical_sha256
from hrvla_bench.recovery_runtime_audit import audit_recovery_runtime_trace


ROOT = Path(__file__).resolve().parents[1]
SUITE = json.loads(
    (ROOT / "benchmark/suites/hrvla_recovery_v0.json").read_text(encoding="utf-8")
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _summary(scenario_id: str, task_id: str, detector_id: str, injector_id: str) -> dict:
    return {
        "schema_version": 1,
        "suite_sha256": canonical_sha256(SUITE),
        "task_id": task_id,
        "scenario_id": scenario_id,
        "episode_seed": 17,
        "triggered": True,
        "trigger_control_step": 8,
        "detector_id": detector_id,
        "injector_id": injector_id,
        "action_samples_modified": 0,
        "initial_snapshot_sha256": None,
        "failure_snapshot_sha256": None,
        "failure_capture_complete": False,
        "runtime_validated": False,
        "claim_boundary": "runtime_validated remains false until an Isaac Sim trace audit passes",
    }


def _reset(scenario_id: str, task_id: str) -> dict:
    return {
        "event": "episode_reset",
        "suite_sha256": canonical_sha256(SUITE),
        "task_id": task_id,
        "scenario_id": scenario_id,
        "episode_seed": 17,
        "control_dt_s": 0.02,
        "environment_index": 0,
        "simulator_revision": "isaac-test",
    }


def _boundary(scenario_id: str, detector_id: str, injector_id: str) -> dict:
    return {
        "event": "semantic_boundary_triggered",
        "scenario_id": scenario_id,
        "episode_seed": 17,
        "control_step": 8,
        "detector_id": detector_id,
        "detector_sample_index": 8,
        "diagnostics": {},
        "signals": {},
        "injector_id": injector_id,
    }


def _result(path: Path, summary: dict) -> Path:
    target = path / "episode.json"
    _write_json(
        target,
        {
            "episode_seed": 17,
            "success": False,
            "failure_reason": "timeout",
            "hrvla_recovery": summary,
        },
    )
    return target


def test_audit_proves_a_numeric_action40_change(tmp_path: Path) -> None:
    scenario = "box-missed-grasp-retry"
    summary = _summary(scenario, "pick_and_place_box", "first-hand-close", "release-grasp-contact")
    summary["action_samples_modified"] = 1
    before = [0.0] * 38 + [1.0, 1.0]
    after = [0.0] * 40
    trace = [
        _reset(scenario, "pick_and_place_box"),
        _boundary(scenario, "first-hand-close", "release-grasp-contact"),
        {
            "event": "action_window_applied",
            "scenario_id": scenario,
            "episode_seed": 17,
            "control_step": 8,
            "injector_id": "release-grasp-contact",
            "elapsed_s": 0.0,
            "shape": [40],
            "input_sha256": canonical_sha256(before),
            "output_sha256": canonical_sha256(after),
            "changed": True,
            "delta_l2": 2.0**0.5,
        },
    ]
    _write_json(tmp_path / "runtime-summary.json", summary)
    _write_jsonl(tmp_path / "runtime-trace.jsonl", trace)
    report = audit_recovery_runtime_trace(SUITE, scenario, tmp_path, _result(tmp_path, summary))
    assert report["runtime_trace_validated"] is True
    assert report["action_samples_modified"] == 1
    assert report["status"] == "runtime_trace_passed_oracle_admission_pending"


def test_audit_rejects_a_self_reported_action_change_without_changed_hashes(
    tmp_path: Path,
) -> None:
    scenario = "box-missed-grasp-retry"
    summary = _summary(scenario, "pick_and_place_box", "first-hand-close", "release-grasp-contact")
    summary["action_samples_modified"] = 1
    action = [0.0] * 40
    trace = [
        _reset(scenario, "pick_and_place_box"),
        _boundary(scenario, "first-hand-close", "release-grasp-contact"),
        {
            "event": "action_window_applied",
            "scenario_id": scenario,
            "episode_seed": 17,
            "control_step": 8,
            "injector_id": "release-grasp-contact",
            "elapsed_s": 0.0,
            "shape": [40],
            "input_sha256": canonical_sha256(action),
            "output_sha256": canonical_sha256(action),
            "changed": True,
            "delta_l2": 1.0,
        },
    ]
    _write_json(tmp_path / "runtime-summary.json", summary)
    _write_jsonl(tmp_path / "runtime-trace.jsonl", trace)
    with pytest.raises(ValueError, match="change flag, hashes, and delta disagree"):
        audit_recovery_runtime_trace(SUITE, scenario, tmp_path, _result(tmp_path, summary))


def test_audit_validates_the_locked_physical_impulse(tmp_path: Path) -> None:
    scenario = "support-state-push"
    summary = _summary(scenario, "football", "object-first-motion", "root-lateral-velocity-delta")
    trace = [
        _reset(scenario, "football"),
        _boundary(scenario, "object-first-motion", "root-lateral-velocity-delta"),
    ]
    audit_row = {
        "event": "apply_root_local_lateral_velocity_once",
        "injector_id": "root-lateral-velocity-delta",
        "asset_name": "robot",
        "environment_ids": [0],
        "episode_seed": 17,
        "episode_steps": [7],
        "lateral_mps": 0.45,
        "lateral_direction_robot": "left",
        "world_velocity_delta": [0.0, 0.45, 0.0],
        "velocity_before": [[0.0] * 6],
        "velocity_after": [[0.0, 0.45, 0.0, 0.0, 0.0, 0.0]],
    }
    _write_json(tmp_path / "runtime-summary.json", summary)
    _write_jsonl(tmp_path / "runtime-trace.jsonl", trace)
    _write_jsonl(tmp_path / "injector-audit.jsonl", [audit_row])
    result = _result(tmp_path, summary)
    report = audit_recovery_runtime_trace(SUITE, scenario, tmp_path, result)
    assert report["runtime_trace_validated"] is True

    wrong = copy.deepcopy(audit_row)
    wrong["world_velocity_delta"] = [0.0, 0.4, 0.0]
    _write_jsonl(tmp_path / "injector-audit.jsonl", [wrong])
    with pytest.raises(ValueError, match="world velocity delta norm"):
        audit_recovery_runtime_trace(SUITE, scenario, tmp_path, result)
