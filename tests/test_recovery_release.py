from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hrvla_bench.evidence import claim_readiness
from hrvla_bench.plan import canonical_sha256
from hrvla_bench.recovery_release import build_admitted_suite


ROOT = Path(__file__).resolve().parents[1]
SUITE = json.loads(
    (ROOT / "benchmark/suites/hrvla_recovery_v0.json").read_text(encoding="utf-8")
)
PREDICATE = json.loads(
    (ROOT / "results/humanoidarena/admission/predicate-source-audit.json").read_text(
        encoding="utf-8"
    )
)
REVISION = "a" * 40


def _capture() -> dict:
    task_rows = []
    scenario_rows = []
    for task_index, task in enumerate(SUITE["tasks"]):
        state_hash = f"{task_index + 1:064x}"
        task_rows.append(
            {
                "task_id": task["id"],
                "snapshot_sha256": state_hash,
                "file_sha256": f"{task_index + 101:064x}",
            }
        )
        for scenario_index, scenario in enumerate(task["scenarios"]):
            failure = scenario["protocol"] == "failure_start"
            scenario_rows.append(
                {
                    "task_id": task["id"],
                    "scenario_id": scenario["id"],
                    "protocol": scenario["protocol"],
                    "runtime_audit_sha256": f"{task_index * 10 + scenario_index + 201:064x}",
                    "implementation_revision": REVISION,
                    "initial_snapshot_sha256": state_hash,
                    "failure_snapshot_sha256": (
                        f"{task_index * 10 + scenario_index + 301:064x}" if failure else None
                    ),
                    "failure_snapshot_file_sha256": (
                        f"{task_index * 10 + scenario_index + 401:064x}" if failure else None
                    ),
                }
            )
    core = {
        "schema_version": 1,
        "status": "runtime_and_snapshots_complete_oracle_admission_pending",
        "suite_id": SUITE["suite_id"],
        "suite_sha256": canonical_sha256(SUITE),
        "implementation_revision": REVISION,
        "initial_snapshots": task_rows,
        "scenarios": scenario_rows,
    }
    return {**core, "manifest_sha256": canonical_sha256(core)}


def _oracles(capture: dict) -> dict[str, dict]:
    reports = {}
    for task in SUITE["tasks"]:
        for scenario in task["scenarios"]:
            core = {
                "status": "admitted",
                "admitted": True,
                "oracle_id": "pi05-independent-oracle",
                "suite_id": SUITE["suite_id"],
                "suite_sha256": canonical_sha256(SUITE),
                "capture_manifest_sha256": capture["manifest_sha256"],
                "scenario_id": scenario["id"],
                "protocol": scenario["protocol"],
                "trials": 20,
                "successes": 20,
                "required_successes": 20,
                "success_rate": 1.0,
            }
            reports[scenario["id"]] = {**core, "audit_sha256": canonical_sha256(core)}
    return reports


def test_complete_evidence_promotes_without_mutating_the_draft() -> None:
    draft = copy.deepcopy(SUITE)
    capture = _capture()
    released = build_admitted_suite(draft, capture, _oracles(capture), PREDICATE)
    assert draft == SUITE
    assert released["replication"]["rollouts_per_seed_per_cell"] == 26
    assert released["admission_provenance"]["draft_suite_sha256"] == canonical_sha256(SUITE)
    assert claim_readiness(released)["claim_ready"] is True
    assert all(task["admission"]["status"] == "admitted" for task in released["tasks"])
    assert all(
        scenario["admission"]["status"] == "admitted"
        for task in released["tasks"]
        for scenario in task["scenarios"]
    )


def test_one_behavioral_oracle_failure_blocks_the_whole_release() -> None:
    capture = _capture()
    reports = _oracles(capture)
    reports["support-state-push"].update(
        status="rejected_behavioral_oracle",
        admitted=False,
        successes=19,
        success_rate=0.95,
    )
    with pytest.raises(ValueError, match="oracle status differs"):
        build_admitted_suite(SUITE, capture, reports, PREDICATE)


def test_mixed_injector_revision_blocks_release() -> None:
    capture = _capture()
    capture["scenarios"][0]["implementation_revision"] = "b" * 40
    reports = _oracles(capture)
    with pytest.raises(ValueError, match="implementation revision differs"):
        build_admitted_suite(SUITE, capture, reports, PREDICATE)
