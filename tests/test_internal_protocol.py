from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hrvla_bench.internal_protocol import (
    build_internal_plans,
    exact_mcnemar_power,
    protocol_summary,
    validate_internal_protocol_lock,
)
from hrvla_bench.plan import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads(
    (ROOT / "config/humanoidarena-internal-protocol.lock.json").read_text(encoding="utf-8")
)


def _admitted_suite() -> dict:
    draft_hash = LOCK["source_suite"]["draft_suite_sha256"]
    suite = {
        "schema_version": "1.0",
        "suite_id": "hrvla_recovery_v0",
        "admission_provenance": {"draft_suite_sha256": draft_hash},
        "controller_contract": {},
        "replication": {"training_seeds": [0, 1, 2], "rollouts_per_seed_per_cell": 20},
        "tasks": [],
    }
    for task_index in range(7):
        scenario_count = 2 if task_index < 2 else 1
        scenarios = []
        for scenario_index in range(scenario_count):
            protocol = "failure_start" if scenario_index else "online_failure"
            scenario = {
                "id": f"scenario-{task_index}-{scenario_index}",
                "protocol": protocol,
                "event_id": "event",
                "event_boundary": "boundary",
                "level": "L1",
                "humanoid_axes": ["H1"],
                "severity": "medium",
                "injector": {"id": "injector", "parameters": {}},
                "admission": {
                    "status": "admitted",
                    "oracle_id": "oracle",
                    "success_rate": 1.0,
                    "oracle_trials": 20,
                    "oracle_successes": 20,
                    "injector_revision": "revision",
                    "predicate_test_id": "test",
                },
            }
            if protocol == "failure_start":
                scenario["failure_snapshot_id"] = f"failure-{task_index}"
                scenario["admission"]["failure_snapshot_sha256"] = "b" * 64
            scenarios.append(scenario)
        suite["tasks"].append(
            {
                "id": f"task-{task_index}",
                "initial_snapshot_id": f"initial-{task_index}",
                "success_predicate": "success",
                "horizon_s": 60,
                "admission": {
                    "status": "admitted",
                    "initial_snapshot_sha256": f"{task_index + 1:064x}",
                    "predicate_test_id": "test",
                },
                "scenarios": scenarios,
            }
        )
    return suite


def test_lock_freezes_disjoint_splits_and_corrected_power() -> None:
    validate_internal_protocol_lock(LOCK)
    power = exact_mcnemar_power(
        78, p_candidate_only=0.26, p_baseline_only=0.05, alpha=0.0125
    )
    assert power == pytest.approx(0.8020639997462751)


def test_internal_plans_have_expected_counts_and_no_seed_overlap() -> None:
    plans = build_internal_plans(_admitted_suite(), LOCK)
    assert {name: len(plan["episodes"]) for name, plan in plans.items()} == {
        "development": 192,
        "validation": 288,
        "hidden_final": 1248,
    }
    summary = protocol_summary(plans, LOCK)
    assert summary["splits"]["hidden_final"]["method_episode_records"] == 6240
    seed_sets = {
        name: {episode["rollout_seed"] for episode in plan["episodes"]}
        for name, plan in plans.items()
    }
    assert seed_sets["development"].isdisjoint(seed_sets["validation"])
    assert seed_sets["development"].isdisjoint(seed_sets["hidden_final"])
    assert seed_sets["validation"].isdisjoint(seed_sets["hidden_final"])


def test_changed_final_replication_fails_power_contract() -> None:
    changed = copy.deepcopy(LOCK)
    changed["splits"]["hidden_final"]["rollouts_per_training_seed_per_cell"] = 20
    changed["splits"]["hidden_final"]["rollout_seeds"] = changed["splits"][
        "hidden_final"
    ]["rollout_seeds"][:20]
    changed["power_analysis"]["paired_observations_per_cell"] = 60
    with pytest.raises(ValueError, match="reported exact McNemar power"):
        validate_internal_protocol_lock(changed)


def test_primary_endpoints_are_bound_before_hidden_results() -> None:
    validate_internal_protocol_lock(LOCK)
    changed = copy.deepcopy(LOCK)
    changed["analysis"]["primary_comparison_endpoints"][1]["protocol"] = "failure_start"
    with pytest.raises(ValueError, match="frozen ablations"):
        validate_internal_protocol_lock(changed)


def test_draft_suite_cannot_materialize_internal_plans() -> None:
    suite = _admitted_suite()
    suite["tasks"][0]["admission"] = {"status": "draft"}
    with pytest.raises(ValueError, match="fully admitted suite"):
        build_internal_plans(suite, LOCK)


def test_plan_hash_changes_if_a_frozen_rollout_seed_changes() -> None:
    suite = _admitted_suite()
    plans = build_internal_plans(suite, LOCK)
    changed_suite = copy.deepcopy(suite)
    changed_suite["admission_provenance"]["unrelated"] = canonical_sha256("change")
    changed = build_internal_plans(changed_suite, LOCK)
    assert plans["hidden_final"]["plan_sha256"] != changed["hidden_final"]["plan_sha256"]
