from __future__ import annotations

import types

import numpy as np

from hrvla_bench.humanoidarena_recovery_signals import (
    _point_box_xy_distance,
    _root_upright,
    signals_for_detector,
)


def test_root_upright_uses_locked_height_and_quaternion_thresholds() -> None:
    state = np.zeros((1, 13), dtype=np.float32)
    state[0, 2] = 0.8
    state[0, 3] = 1.0
    robot = types.SimpleNamespace(data=types.SimpleNamespace(root_state_w=state))
    env = types.SimpleNamespace(scene={"robot": robot})
    upright, height, up_axis_z = _root_upright(
        env, minimum_root_height_m=0.45, minimum_up_axis_z=0.6
    )
    assert upright is True
    assert np.isclose(height, 0.8)
    assert np.isclose(up_axis_z, 1.0)

    state[0, 3:7] = [0.5, 0.8660254, 0.0, 0.0]
    upright, _, up_axis_z = _root_upright(
        env, minimum_root_height_m=0.45, minimum_up_axis_z=0.6
    )
    assert upright is False
    assert up_axis_z < 0.0


def test_point_to_live_seat_aabb_distance_is_zero_inside() -> None:
    box = (1.0, 2.0, 3.0, 5.0, 0.2, 1.0)
    assert _point_box_xy_distance((1.5, 4.0), box) == 0.0
    assert np.isclose(_point_box_xy_distance((0.4, 2.2), box), 1.0)


def test_scene_only_detector_does_not_request_private_task_signals() -> None:
    assert signals_for_detector("root-displacement", object(), {}) == {}
