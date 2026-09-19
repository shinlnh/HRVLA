"""Promote a draft recovery suite only from complete audited admission evidence."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .plan import canonical_sha256, validate_suite


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value) - set("0123456789abcdef")
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _index(rows: Any, key: str, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be a list")
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key) if isinstance(row, dict) else None
        if not isinstance(value, str) or not value or value in output:
            raise ValueError(f"{label} contains a missing or duplicate {key}")
        output[value] = row
    return output


def build_admitted_suite(
    draft_suite: dict[str, Any],
    capture_manifest: dict[str, Any],
    oracle_reports: Mapping[str, dict[str, Any]],
    predicate_audit: dict[str, Any],
) -> dict[str, Any]:
    """Return a new admitted suite; never mutate or overwrite the draft contract."""

    validate_suite(draft_suite)
    draft_sha = canonical_sha256(draft_suite)
    suite_id = draft_suite["suite_id"]
    if any(
        task.get("admission", {}).get("status") != "draft"
        or any(
            scenario.get("admission", {}).get("status") != "draft"
            for scenario in task.get("scenarios", [])
        )
        for task in draft_suite["tasks"]
    ):
        raise ValueError("release compiler requires the immutable all-draft source suite")

    if capture_manifest.get("status") != "runtime_and_snapshots_complete_oracle_admission_pending":
        raise ValueError("capture manifest is not runtime/snapshot complete")
    if capture_manifest.get("suite_id") != suite_id:
        raise ValueError("capture manifest suite ID differs")
    if capture_manifest.get("suite_sha256") != draft_sha:
        raise ValueError("capture manifest suite hash differs")
    capture_sha = _sha256(capture_manifest.get("manifest_sha256"), "capture manifest hash")
    implementation_revision = capture_manifest.get("implementation_revision")
    if (
        not isinstance(implementation_revision, str)
        or len(implementation_revision) != 40
        or set(implementation_revision) - set("0123456789abcdef")
    ):
        raise ValueError("capture manifest implementation revision is malformed")

    if predicate_audit.get("suite_id") != suite_id:
        raise ValueError("predicate audit suite ID differs")
    if predicate_audit.get("suite_sha256") != draft_sha:
        raise ValueError("predicate audit suite hash differs")
    predicate_tasks = _index(predicate_audit.get("tasks"), "task_id", "predicate tasks")
    capture_tasks = _index(
        capture_manifest.get("initial_snapshots"), "task_id", "capture tasks"
    )
    capture_scenarios = _index(
        capture_manifest.get("scenarios"), "scenario_id", "capture scenarios"
    )
    expected_tasks = {task["id"] for task in draft_suite["tasks"]}
    expected_scenarios = {
        scenario["id"]
        for task in draft_suite["tasks"]
        for scenario in task["scenarios"]
    }
    if set(predicate_tasks) != expected_tasks or set(capture_tasks) != expected_tasks:
        raise ValueError("predicate/capture task membership differs from the draft suite")
    if set(capture_scenarios) != expected_scenarios:
        raise ValueError("capture scenario membership differs from the draft suite")
    if set(oracle_reports) != expected_scenarios:
        raise ValueError("oracle report membership differs from the draft suite")

    released = deepcopy(draft_suite)
    released["status"] = "admitted_for_internal_plan_open_door_diagnostic"
    released["replication"]["rollouts_per_seed_per_cell"] = 26
    released["admission_provenance"] = {
        "draft_suite_sha256": draft_sha,
        "capture_manifest_sha256": capture_sha,
        "predicate_audit_sha256": canonical_sha256(predicate_audit),
        "implementation_revision": implementation_revision,
        "open_door_primary_aggregate_eligible": False,
    }
    released["reference_stack"]["vla"] = (
        "GR00T N1.7 HumanoidArena state64/action40 checkpoints bound per method and training seed"
    )

    for task in released["tasks"]:
        task_id = task["id"]
        initial = capture_tasks[task_id]
        predicate = predicate_tasks[task_id]
        initial_state_sha = _sha256(initial.get("snapshot_sha256"), "initial snapshot hash")
        initial_file_sha = _sha256(initial.get("file_sha256"), "initial snapshot file hash")
        predicate_source = predicate.get("predicate_source", {})
        predicate_source_sha = _sha256(
            predicate_source.get("source_sha256"), "predicate source hash"
        )
        if predicate.get("predicate") != task.get("success_predicate"):
            raise ValueError(f"{task_id}: audited predicate text differs from the suite")
        task["initial_snapshot_id"] = f"sha256:{initial_state_sha}"
        task["admission"] = {
            "status": "admitted",
            "initial_snapshot_sha256": initial_state_sha,
            "initial_snapshot_file_sha256": initial_file_sha,
            "predicate_test_id": (
                "results/humanoidarena/admission/predicate-source-audit.json#"
                f"{task_id}@sha256:{predicate_source_sha}"
            ),
        }
        for scenario in task["scenarios"]:
            scenario_id = scenario["id"]
            capture = capture_scenarios[scenario_id]
            oracle = oracle_reports[scenario_id]
            expected = {
                "status": "admitted",
                "admitted": True,
                "suite_id": suite_id,
                "suite_sha256": draft_sha,
                "capture_manifest_sha256": capture_sha,
                "scenario_id": scenario_id,
                "protocol": scenario["protocol"],
                "trials": 20,
                "successes": 20,
                "required_successes": 20,
                "success_rate": 1.0,
            }
            for key, value in expected.items():
                if oracle.get(key) != value:
                    raise ValueError(f"{scenario_id}: oracle {key} differs")
            oracle_audit_sha = _sha256(oracle.get("audit_sha256"), "oracle audit hash")
            runtime_audit_sha = _sha256(
                capture.get("runtime_audit_sha256"), "runtime audit hash"
            )
            if capture.get("task_id") != task_id or capture.get("protocol") != scenario["protocol"]:
                raise ValueError(f"{scenario_id}: capture identity differs")
            if capture.get("implementation_revision") != implementation_revision:
                raise ValueError(f"{scenario_id}: capture implementation revision differs")
            if capture.get("initial_snapshot_sha256") != initial_state_sha:
                raise ValueError(f"{scenario_id}: capture initial snapshot differs")

            admission = {
                "status": "admitted",
                "oracle_id": oracle.get("oracle_id"),
                "success_rate": 1.0,
                "oracle_trials": 20,
                "oracle_successes": 20,
                "oracle_audit_sha256": oracle_audit_sha,
                "injector_revision": implementation_revision,
                "runtime_audit_sha256": runtime_audit_sha,
                "predicate_test_id": f"runtime-audit:sha256:{runtime_audit_sha}",
            }
            if not isinstance(admission["oracle_id"], str) or not admission["oracle_id"]:
                raise ValueError(f"{scenario_id}: oracle ID is missing")
            if scenario["protocol"] == "failure_start":
                failure_sha = _sha256(
                    capture.get("failure_snapshot_sha256"), "failure snapshot hash"
                )
                failure_file_sha = _sha256(
                    capture.get("failure_snapshot_file_sha256"),
                    "failure snapshot file hash",
                )
                scenario["failure_snapshot_id"] = f"sha256:{failure_sha}"
                admission["failure_snapshot_sha256"] = failure_sha
                admission["failure_snapshot_file_sha256"] = failure_file_sha
            elif capture.get("failure_snapshot_sha256") is not None:
                raise ValueError(f"{scenario_id}: online scenario has a failure snapshot")
            scenario["admission"] = admission

    validate_suite(released)
    return released


__all__ = ["build_admitted_suite"]
