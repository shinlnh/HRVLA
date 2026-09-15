from __future__ import annotations

import numpy as np
import pytest

from hrvla_bench.subtask_relabel import segment_subtasks


def _episode(length: int = 240) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((length, 64), dtype=np.float32), np.zeros((length, 40), dtype=np.float32)


def test_manipulation_labels_follow_close_then_release_monotonically() -> None:
    state, action = _episode()
    action[70:170, 39] = 1.0
    segmented = segment_subtasks("pp_box", state, action, skill_count=3)
    assert segmented.audit["boundaries"] == [70, 170]
    assert segmented.audit["fallback_used"] is False
    np.testing.assert_array_equal(np.unique(segmented.labels), [0, 1, 2])


def test_navigation_uses_integrated_reference_displacement() -> None:
    state, action = _episode()
    action[:, 0] = 0.01
    segmented = segment_subtasks("vision_navi", state, action, skill_count=2)
    assert 99 <= segmented.audit["boundaries"][0] <= 100
    assert segmented.audit["fallback_used"] is False


def test_missing_observable_is_retained_as_an_explicit_weak_fallback() -> None:
    state, action = _episode()
    segmented = segment_subtasks("open_door", state, action, skill_count=3)
    assert segmented.audit["fallback_used"] is True
    assert segmented.audit["boundaries"] == [80, 160]


def test_ordered_change_point_proxy_segments_motion_when_hand_signal_is_absent() -> None:
    state, action = _episode()
    state[:80, 35] = np.tile([0.0, 1.0], 40)
    state[80:160, 49] = np.tile([0.0, 3.0], 40)
    action[160:, 0] = np.tile([0.0, 0.08], 40)
    segmented = segment_subtasks("open_door", state, action, skill_count=3)
    assert segmented.audit["proxy"] == "ordered-multivariate-change-point"
    assert segmented.audit["fallback_used"] is False
    first, second = segmented.audit["boundaries"]
    assert 60 <= first <= 100
    assert 140 <= second <= 180


def test_subtask_relabel_rejects_the_old_43_dof_contract() -> None:
    state, _action = _episode()
    with pytest.raises(ValueError, match=r"\[N, 40\]"):
        segment_subtasks("boxing", state, np.zeros((240, 43)), skill_count=2)
