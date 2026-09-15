from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hrvla_bench.recovery_oracle import evaluate_oracle_trials, validate_oracle_lock


ROOT = Path(__file__).resolve().parents[1]
SUITE = json.loads(
    (ROOT / "benchmark/suites/hrvla_recovery_v0.json").read_text(encoding="utf-8")
)
LOCK = json.loads(
    (ROOT / "config/humanoidarena-recovery-oracle.lock.json").read_text(
        encoding="utf-8"
    )
)


def _manifest() -> dict:
    scenarios = []
    for task in SUITE["tasks"]:
        for scenario in task["scenarios"]:
            scenarios.append(
                {
                    "task_id": task["id"],
                    "scenario_id": scenario["id"],
                    "initial_snapshot_sha256": "a" * 64,
                    "failure_snapshot_sha256": (
                        "b" * 64 if scenario["protocol"] == "failure_start" else None
                    ),
                }
            )
    return {
        "suite_sha256": LOCK["suite_sha256"],
        "manifest_sha256": "c" * 64,
        "scenarios": scenarios,
    }


def _records(scenario_id: str, *, successes: int = 20) -> list[dict]:
    scenario = next(
        scenario
        for task in SUITE["tasks"]
        for scenario in task["scenarios"]
        if scenario["id"] == scenario_id
    )
    independence = LOCK["independence"]
    records = []
    for index, seed in enumerate(LOCK["rollout_seeds"][scenario_id]):
        success = index < successes
        row = {
            "oracle_id": LOCK["oracle_id"],
            "suite_id": SUITE["suite_id"],
            "suite_sha256": LOCK["suite_sha256"],
            "capture_manifest_sha256": "c" * 64,
            "scenario_id": scenario_id,
            "protocol": scenario["protocol"],
            "trial_index": index,
            "rollout_seed": seed,
            "model_revision": independence["model_revision"],
            "source_revision": independence["source_revision"],
            "isaac_lab_revision": independence["isaac_lab_revision"],
            "sonic_revision": independence["sonic_revision"],
            "initial_snapshot_sha256": "a" * 64,
            "failure_snapshot_sha256": (
                "b" * 64 if scenario["protocol"] == "failure_start" else None
            ),
            "success": success,
            "failure_reason": "success" if success else "timeout",
            "episode_result_sha256": f"{index + 1:064x}",
        }
        evidence_key = (
            "failure_start_trial_audit_sha256"
            if scenario["protocol"] == "failure_start"
            else "runtime_audit_sha256"
        )
        row["start_state_restore_audit_sha256"] = f"{index + 201:064x}"
        row[evidence_key] = f"{index + 101:064x}"
        records.append(row)
    return records


def test_oracle_lock_has_twenty_unique_predeclared_seeds_per_scenario() -> None:
    validate_oracle_lock(LOCK, SUITE)


@pytest.mark.parametrize(
    "scenario_id", ["support-state-push", "box-drop-and-body-push"]
)
def test_oracle_admits_only_a_complete_twenty_of_twenty(scenario_id: str) -> None:
    report = evaluate_oracle_trials(
        LOCK, SUITE, _manifest(), scenario_id, _records(scenario_id)
    )
    assert report["admitted"] is True
    assert report["trials"] == report["successes"] == 20


def test_behavioral_failure_rejects_without_discarding_the_trial() -> None:
    scenario = "support-state-push"
    report = evaluate_oracle_trials(
        LOCK, SUITE, _manifest(), scenario, _records(scenario, successes=19)
    )
    assert report["admitted"] is False
    assert report["status"] == "rejected_behavioral_oracle"
    assert report["result_reason_counts"] == {"success": 19, "timeout": 1}


def test_infrastructure_failure_cannot_be_counted_as_an_oracle_trial() -> None:
    scenario = "support-state-push"
    records = copy.deepcopy(_records(scenario))
    records[0].update(success=False, failure_reason="sim_error")
    with pytest.raises(ValueError, match="infrastructure failure"):
        evaluate_oracle_trials(LOCK, SUITE, _manifest(), scenario, records)
