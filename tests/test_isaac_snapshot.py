from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from hrvla_bench.isaac_snapshot import (
    capture_scene_snapshot,
    load_snapshot,
    validate_snapshot,
    write_snapshot_atomic,
)


class Data:
    def __init__(self, root, joint_pos=None, joint_vel=None):
        self.root_state_w = np.asarray(root, dtype=np.float32)
        if joint_pos is not None:
            self.joint_pos = np.asarray(joint_pos, dtype=np.float32)
            self.joint_vel = np.asarray(joint_vel, dtype=np.float32)


class Asset:
    def __init__(self, root, joints=0):
        positions = [[0.1 * index for index in range(joints)]] if joints else None
        velocities = [[0.0] * joints] if joints else None
        self.data = Data(root, positions, velocities)
        self.joint_names = [f"joint_{index}" for index in range(joints)]


class Env:
    def __init__(self):
        self.num_envs = 1
        self.episode_length_buf = np.asarray([123])
        self.scene = {
            "robot": Asset([[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], joints=3),
            "object": Asset([[1, 2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        }


def test_snapshot_is_deterministic_hash_addressed_and_roundtrips(tmp_path: Path) -> None:
    kwargs = {
        "task_id": "pick_and_place_box",
        "event_id": "after-stable-grasp",
        "episode_seed": 17,
        "scene_assets": ["object", "robot"],
        "simulator_revision": "isaac@test",
    }
    first = capture_scene_snapshot(Env(), **kwargs)
    second = capture_scene_snapshot(Env(), **kwargs)
    assert first == second
    assert list(first["assets"]) == ["object", "robot"]
    assert len(validate_snapshot(first)) == 64
    path = tmp_path / "snapshot.json"
    write_snapshot_atomic(path, first)
    assert load_snapshot(path) == first
    assert json.loads(path.read_text()) == first


def test_snapshot_detects_state_or_provenance_tampering() -> None:
    snapshot = capture_scene_snapshot(
        Env(),
        task_id="boxing",
        event_id="contact",
        episode_seed=1,
        scene_assets=["robot"],
        simulator_revision="isaac@test",
    )
    changed = copy.deepcopy(snapshot)
    changed["assets"]["robot"]["root_state_w"][0] = 99
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_snapshot(changed)
    changed = copy.deepcopy(snapshot)
    changed["episode_seed"] = 2
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_snapshot(changed)
