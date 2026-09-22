"""Recovery routing consumes a declared observed failure, never a guessed one."""

from pathlib import Path

import pytest

from hrvla_subtask.http_routing import RecoveryObservedStateRouter


ROOT = Path(__file__).resolve().parents[1]


def test_recovery_selects_bounded_handoff_instruction():
    router = RecoveryObservedStateRouter.from_paths(
        ROOT / "benchmark/subtask_suite_v1.json",
        ROOT / "benchmark/recovery_protocol_v1.json",
    )
    instruction, primitive_id = router.select("fruit_bowl_and_drawer", {
        "failure_active": True,
        "failure_label": "dropped_object",
        "observed_predicates": ["apple_on_floor", "robot_at_table"],
        "pre_failure_predicates": ["holding_apple", "robot_at_table"],
    })
    assert primitive_id == "transition_regrasp_apple"
    assert "wrist above the bowl" in instruction


def test_recovery_rejects_missing_observation_and_wrong_label():
    router = RecoveryObservedStateRouter.from_paths(
        ROOT / "benchmark/subtask_suite_v1.json",
        ROOT / "benchmark/recovery_protocol_v1.json",
    )
    with pytest.raises(ValueError, match="failure label"):
        router.select("fruit_bowl_and_drawer", {
            "failure_active": True, "failure_label": "fabricated",
        })
    with pytest.raises(ValueError, match="pre_failure_predicates"):
        router.select("fruit_bowl_and_drawer", {
            "failure_active": True, "failure_label": "dropped_object",
            "observed_predicates": ["apple_on_floor"],
        })
