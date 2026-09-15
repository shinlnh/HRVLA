from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hrvla_bench.plan import canonical_sha256
from hrvla_bench.recovery_admission import build_capture_manifest


ROOT = Path(__file__).resolve().parents[1]
SUITE = json.loads(
    (ROOT / "benchmark/suites/hrvla_recovery_v0.json").read_text(encoding="utf-8")
)


def _capture_rows() -> dict[str, dict]:
    suite_sha = canonical_sha256(SUITE)
    initial_by_task = {
        task["id"]: f"{index + 1:064x}" for index, task in enumerate(SUITE["tasks"])
    }
    rows = {}
    failure_index = 100
    for task in SUITE["tasks"]:
        for scenario in task["scenarios"]:
            failure = None
            if scenario["protocol"] == "failure_start":
                failure_index += 1
                failure = {
                    "snapshot_sha256": f"{failure_index:064x}",
                    "file_sha256": f"{failure_index + 100:064x}",
                    "path": "failure.json",
                }
            initial_hash = initial_by_task[task["id"]]
            audit_core = {
                "suite_sha256": suite_sha,
                "task_id": task["id"],
                "scenario_id": scenario["id"],
                "implementation_revision": "a" * 40,
                "runtime_trace_validated": True,
                "snapshots": {
                    "initial": {
                        "snapshot_sha256": initial_hash,
                        "file_sha256": initial_hash,
                        "path": "initial.json",
                    },
                    **({} if failure is None else {"failure": failure}),
                },
            }
            audit_core["audit_sha256"] = canonical_sha256(audit_core)
            rows[scenario["id"]] = {
                "audit": audit_core,
                "evidence_directory": f"{scenario['id']}/attempt-0001",
            }
    return rows


def test_complete_capture_manifest_locks_nine_seven_and_three() -> None:
    manifest = build_capture_manifest(SUITE, _capture_rows())
    assert manifest["gates"]["runtime_injectors"] == {"complete": 9, "required": 9}
    assert manifest["gates"]["initial_snapshots"] == {"complete": 7, "required": 7}
    assert manifest["gates"]["failure_snapshots"] == {"complete": 3, "required": 3}
    assert manifest["gates"]["oracle_20_of_20"] == {"complete": 0, "required": 9}
    assert len(manifest["manifest_sha256"]) == 64
    assert manifest["implementation_revision"] == "a" * 40


def test_capture_manifest_rejects_initial_state_drift_within_a_task() -> None:
    captures = _capture_rows()
    changed = copy.deepcopy(captures)
    changed["box-drop-and-body-push"]["audit"]["snapshots"]["initial"][
        "snapshot_sha256"
    ] = "f" * 64
    with pytest.raises(ValueError, match="initial snapshots differ"):
        build_capture_manifest(SUITE, changed)


def test_capture_manifest_rejects_a_missing_scenario() -> None:
    captures = _capture_rows()
    del captures["support-state-push"]
    with pytest.raises(ValueError, match="capture set differs"):
        build_capture_manifest(SUITE, captures)
