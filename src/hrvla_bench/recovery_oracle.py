"""Locked independent-oracle contract for recovery scenario admission."""

from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any, Iterable

from .plan import canonical_sha256


INFRASTRUCTURE_FAILURES = {
    "interrupted",
    "process_error",
    "sim_error",
    "sim_stopped",
    "unknown",
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _is_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and not (set(value) - set("0123456789abcdef"))
    )


def _derive_seed(scenario_id: str, trial_index: int) -> int:
    payload = f"hrvla-oracle-v0|{scenario_id}|{trial_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[-4:], "big") & 0x7FFFFFFF


def validate_oracle_lock(lock: dict[str, Any], suite: dict[str, Any]) -> None:
    if lock.get("schema_version") != 1:
        raise ValueError("oracle lock schema_version must be 1")
    if lock.get("suite_id") != suite.get("suite_id"):
        raise ValueError("oracle lock suite ID differs")
    suite_sha256 = canonical_sha256(suite)
    if lock.get("suite_sha256") != suite_sha256:
        raise ValueError("oracle lock suite hash differs")
    if not str(lock.get("oracle_id", "")).strip():
        raise ValueError("oracle_id is required")
    independence = lock.get("independence", {})
    if independence.get("oracle_model_family") == independence.get(
        "evaluated_internal_model_family"
    ):
        raise ValueError("admission oracle must use an independent model family")
    for key in (
        "model_revision",
        "source_revision",
        "isaac_lab_revision",
        "sonic_revision",
    ):
        if not _is_revision(independence.get(key)):
            raise ValueError(f"oracle independence {key} must be a content revision")
    protocol = lock.get("protocol", {})
    required_trials = protocol.get("required_trials")
    required_successes = protocol.get("required_successes")
    if required_trials != 20 or required_successes != required_trials:
        raise ValueError("recovery admission requires a predeclared 20/20 oracle")
    if protocol.get("policy_seed_is_the_only_varied_factor") is not True:
        raise ValueError("oracle trials must vary only the locked policy seed")

    scenario_ids = {
        scenario["id"] for task in suite["tasks"] for scenario in task["scenarios"]
    }
    rollout_seeds = lock.get("rollout_seeds")
    if not isinstance(rollout_seeds, dict) or set(rollout_seeds) != scenario_ids:
        raise ValueError("oracle rollout-seed scenarios differ from the suite")
    for scenario_id, seeds in rollout_seeds.items():
        if (
            not isinstance(seeds, list)
            or len(seeds) != required_trials
            or len(set(seeds)) != required_trials
            or any(type(seed) is not int or seed < 0 for seed in seeds)
        ):
            raise ValueError(f"{scenario_id}: oracle requires 20 unique non-negative seeds")
        expected_seeds = [_derive_seed(scenario_id, index) for index in range(required_trials)]
        if seeds != expected_seeds:
            raise ValueError(f"{scenario_id}: oracle seeds differ from the locked derivation")


def evaluate_oracle_trials(
    lock: dict[str, Any],
    suite: dict[str, Any],
    capture_manifest: dict[str, Any],
    scenario_id: str,
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Validate exactly one behavioral result for every frozen oracle seed."""

    validate_oracle_lock(lock, suite)
    if capture_manifest.get("suite_sha256") != canonical_sha256(suite):
        raise ValueError("capture manifest suite hash differs")
    capture_manifest_sha = capture_manifest.get("manifest_sha256")
    if not _is_sha256(capture_manifest_sha):
        raise ValueError("capture manifest hash is malformed")
    scenario = next(
        (
            scenario
            for task in suite["tasks"]
            for scenario in task["scenarios"]
            if scenario["id"] == scenario_id
        ),
        None,
    )
    if scenario is None:
        raise ValueError(f"unknown recovery scenario: {scenario_id}")
    capture = next(
        row for row in capture_manifest["scenarios"] if row["scenario_id"] == scenario_id
    )
    expected_seeds = list(lock["rollout_seeds"][scenario_id])
    rows = list(records)
    if len(rows) != len(expected_seeds):
        raise ValueError(
            f"{scenario_id}: expected {len(expected_seeds)} oracle records, got {len(rows)}"
        )
    by_index = Counter(int(row.get("trial_index", -1)) for row in rows)
    if set(by_index) != set(range(len(expected_seeds))) or any(
        count != 1 for count in by_index.values()
    ):
        raise ValueError(f"{scenario_id}: oracle trial indices are missing or duplicated")

    ordered = sorted(rows, key=lambda row: int(row["trial_index"]))
    independence = lock["independence"]
    for index, row in enumerate(ordered):
        expected = {
            "oracle_id": lock["oracle_id"],
            "suite_id": suite["suite_id"],
            "suite_sha256": canonical_sha256(suite),
            "capture_manifest_sha256": capture_manifest_sha,
            "scenario_id": scenario_id,
            "protocol": scenario["protocol"],
            "trial_index": index,
            "rollout_seed": expected_seeds[index],
            "model_revision": independence["model_revision"],
            "source_revision": independence["source_revision"],
            "isaac_lab_revision": independence["isaac_lab_revision"],
            "sonic_revision": independence["sonic_revision"],
            "initial_snapshot_sha256": capture["initial_snapshot_sha256"],
            "failure_snapshot_sha256": capture["failure_snapshot_sha256"],
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f"{scenario_id}/trial-{index}: {key} differs")
        if row.get("failure_reason") in INFRASTRUCTURE_FAILURES:
            raise ValueError(
                f"{scenario_id}/trial-{index}: infrastructure failure is not an oracle trial"
            )
        if not isinstance(row.get("success"), bool):
            raise ValueError(f"{scenario_id}/trial-{index}: success must be boolean")
        if row["success"] != (row.get("failure_reason") == "success"):
            raise ValueError(f"{scenario_id}/trial-{index}: success/reason mismatch")
        protocol_evidence_key = (
            "failure_start_trial_audit_sha256"
            if scenario["protocol"] == "failure_start"
            else "runtime_audit_sha256"
        )
        for key in (
            "episode_result_sha256",
            "start_state_restore_audit_sha256",
            protocol_evidence_key,
        ):
            if not _is_sha256(row.get(key)):
                raise ValueError(f"{scenario_id}/trial-{index}: {key} is malformed")

    successes = sum(row["success"] for row in ordered)
    required = int(lock["protocol"]["required_successes"])
    core = {
        "schema_version": 1,
        "status": "admitted" if successes == required else "rejected_behavioral_oracle",
        "claim_boundary": "independent recoverability witness only; not a method score",
        "oracle_id": lock["oracle_id"],
        "suite_id": suite["suite_id"],
        "suite_sha256": canonical_sha256(suite),
        "capture_manifest_sha256": capture_manifest_sha,
        "scenario_id": scenario_id,
        "protocol": scenario["protocol"],
        "trials": len(ordered),
        "successes": successes,
        "required_successes": required,
        "success_rate": successes / len(ordered),
        "admitted": successes == required,
        "rollout_seeds": expected_seeds,
        "result_reason_counts": dict(
            sorted(Counter(str(row["failure_reason"]) for row in ordered).items())
        ),
        "episode_result_sha256": [row["episode_result_sha256"] for row in ordered],
    }
    return {**core, "audit_sha256": canonical_sha256(core)}


__all__ = ["evaluate_oracle_trials", "validate_oracle_lock"]
