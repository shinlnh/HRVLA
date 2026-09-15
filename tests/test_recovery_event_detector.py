from __future__ import annotations

import types

import numpy as np
import pytest

from hrvla_bench.recovery_event_detector import SemanticEventDetector


class _Asset:
    def __init__(self) -> None:
        pose = np.zeros((1, 7), dtype=np.float32)
        pose[:, 3] = 1.0
        self.data = types.SimpleNamespace(
            root_link_pose_w=pose,
            root_vel_w=np.zeros((1, 6), dtype=np.float32),
        )


class _Env:
    def __init__(self) -> None:
        self.scene = {
            "robot": _Asset(),
            "box": _Asset(),
            "object_l": _Asset(),
            "object": _Asset(),
        }


def _detector(detector_id: str, parameters: dict, env: _Env) -> SemanticEventDetector:
    detector = SemanticEventDetector(
        {"id": detector_id, "parameters": parameters}, control_dt_s=0.1
    )
    detector.reset(env)
    return detector


def test_first_hand_close_uses_one_latched_rising_edge() -> None:
    env = _Env()
    detector = _detector(
        "first-hand-close", {"close_threshold": 0.5, "hand_indices": [38, 39]}, env
    )
    action = np.zeros(40, dtype=np.float32)
    assert not detector.observe(env, task_success=False, semantic_action=action).triggered
    action[39] = 0.5
    first = detector.observe(env, task_success=False, semantic_action=action)
    assert first.triggered and first.diagnostics["hand_values"] == [0.0, 0.5]
    assert not detector.observe(env, task_success=False, semantic_action=action).triggered


def test_task_success_prevents_a_late_failure_trigger() -> None:
    env = _Env()
    detector = _detector(
        "root-displacement", {"minimum_xy_displacement_m": 1.0}, env
    )
    env.scene["robot"].data.root_link_pose_w[0, 0] = 2.0
    observation = detector.observe(env, task_success=True)
    assert not observation.triggered
    assert observation.diagnostics["task_success"] is True


def test_stable_lift_requires_the_full_contiguous_duration() -> None:
    env = _Env()
    detector = _detector(
        "stable-object-lift",
        {
            "asset_name": "box",
            "minimum_lift_m": 0.05,
            "maximum_speed_mps": 0.08,
            "stable_duration_s": 0.25,
        },
        env,
    )
    env.scene["box"].data.root_link_pose_w[0, 2] = 0.06
    for _ in range(2):
        assert not detector.observe(env, task_success=False).triggered
    assert detector.observe(env, task_success=False).triggered


@pytest.mark.parametrize(
    ("detector_id", "parameters", "mutate", "diagnostic"),
    [
        (
            "object-in-transport",
            {
                "asset_name": "object_l",
                "minimum_displacement_m": 0.2,
                "minimum_lift_m": 0.05,
            },
            lambda env: env.scene["object_l"].data.root_link_pose_w.__setitem__(
                (0, slice(0, 3)), [0.2, 0.0, 0.06]
            ),
            "displacement_m",
        ),
        (
            "object-first-motion",
            {"asset_name": "object", "minimum_speed_mps": 0.25},
            lambda env: env.scene["object"].data.root_vel_w.__setitem__(
                (0, slice(0, 3)), [0.0, 0.3, 0.0]
            ),
            "speed_mps",
        ),
        (
            "root-displacement",
            {"minimum_xy_displacement_m": 1.0},
            lambda env: env.scene["robot"].data.root_link_pose_w.__setitem__(
                (0, slice(0, 2)), [0.8, 0.6]
            ),
            "xy_displacement_m",
        ),
    ],
)
def test_scene_state_detectors(detector_id, parameters, mutate, diagnostic) -> None:
    env = _Env()
    detector = _detector(detector_id, parameters, env)
    mutate(env)
    observation = detector.observe(env, task_success=False)
    assert observation.triggered
    assert diagnostic in observation.diagnostics


@pytest.mark.parametrize(
    ("detector_id", "parameters", "signals"),
    [
        (
            "wrist-handle-approach",
            {"door_asset_name": "door", "maximum_distance_m": 0.3},
            {"wrist_handle_min_distance_m": 0.29},
        ),
        (
            "door-open-before-traversal",
            {"minimum_leaf_angle_deg": 60.0},
            {"door_leaf_angle_deg": -61.0, "door_latch_unlocked": True},
        ),
        (
            "seat-approach",
            {
                "maximum_xy_distance_m": 0.75,
                "minimum_root_height_m": 0.45,
                "minimum_up_axis_z": 0.6,
            },
            {"seat_xy_distance_m": 0.7, "robot_upright": True},
        ),
    ],
)
def test_external_geometry_signal_detectors(detector_id, parameters, signals) -> None:
    env = _Env()
    detector = _detector(detector_id, parameters, env)
    assert detector.observe(env, task_success=False, signals=signals).triggered


def test_strike_shell_requires_strictly_decreasing_clearance() -> None:
    env = _Env()
    detector = _detector(
        "strike-approach-shell",
        {"inner_margin_m": 0.08, "outer_margin_m": 0.2, "decreasing_samples": 3},
        env,
    )
    for clearance in (0.30, 0.21):
        assert not detector.observe(
            env,
            task_success=False,
            signals={"punch_hit_clearance_m": clearance},
        ).triggered
    result = detector.observe(
        env, task_success=False, signals={"punch_hit_clearance_m": 0.19}
    )
    assert result.triggered
    assert result.diagnostics["strictly_decreasing"] is True


def test_external_detector_rejects_missing_or_nonfinite_signal() -> None:
    env = _Env()
    detector = _detector(
        "seat-approach",
        {
            "maximum_xy_distance_m": 0.75,
            "minimum_root_height_m": 0.45,
            "minimum_up_axis_z": 0.6,
        },
        env,
    )
    with pytest.raises(ValueError, match="seat_xy_distance_m"):
        detector.observe(env, task_success=False, signals={"robot_upright": True})
