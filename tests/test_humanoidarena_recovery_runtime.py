from __future__ import annotations

import json
from pathlib import Path
import types

import numpy as np
import pytest

from hrvla_bench.humanoidarena_recovery_runtime import HumanoidArenaRecoveryRuntime


ROOT = Path(__file__).resolve().parents[1]
SUITE = json.loads(
    (ROOT / "benchmark/suites/hrvla_recovery_v0.json").read_text(encoding="utf-8")
)


class _Asset:
    def __init__(self, torch_module) -> None:
        self.device = torch_module.device("cpu")
        pose = torch_module.zeros((1, 7))
        pose[:, 3] = 1.0
        self.body_names = ["torso_link", "left_ankle_roll_link", "right_ankle_roll_link"]
        self.data = types.SimpleNamespace(
            root_link_pose_w=pose,
            root_state_w=torch_module.cat((pose, torch_module.zeros((1, 6))), dim=1),
            root_vel_w=torch_module.zeros((1, 6)),
            body_names=self.body_names,
        )

    def write_root_velocity_to_sim(self, values, *, env_ids) -> None:
        self.data.root_vel_w[env_ids] = values

    def write_root_pose_to_sim(self, values, *, env_ids) -> None:
        self.data.root_link_pose_w[env_ids] = values

    def set_external_force_and_torque(self, *args, **kwargs) -> None:
        pass


class _Scene(dict):
    def __init__(self, torch_module) -> None:
        super().__init__(
            robot=_Asset(torch_module),
            box=_Asset(torch_module),
            object=_Asset(torch_module),
        )
        self.num_envs = 1


class _Env:
    def __init__(self, torch_module) -> None:
        self.device = torch_module.device("cpu")
        self.num_envs = 1
        self.scene = _Scene(torch_module)
        self.episode_length_buf = torch_module.zeros(1, dtype=torch_module.long)


def test_runtime_triggers_and_modifies_the_first_close_action(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    env = _Env(torch)
    runtime = HumanoidArenaRecoveryRuntime(
        SUITE,
        "box-missed-grasp-retry",
        control_dt_s=0.02,
        simulator_revision="isaac-test",
        output_dir=tmp_path,
    )
    runtime.reset(env, episode_seed=7)
    runtime.before_control_step(env, task_success=False)
    action = np.zeros(40, dtype=np.float32)
    action[38:] = 1.0
    modified = runtime.transform_vla_action(env, action, task_success=False)
    assert runtime.triggered
    np.testing.assert_array_equal(modified[38:], [0.0, 0.0])
    assert runtime.summary()["action_samples_modified"] == 1
    assert runtime.summary()["runtime_validated"] is False
    events = [json.loads(line)["event"] for line in runtime.trace_path.read_text().splitlines()]
    assert events == [
        "episode_reset",
        "semantic_boundary_triggered",
        "action_window_applied",
    ]
    action_event = json.loads(runtime.trace_path.read_text().splitlines()[-1])
    assert action_event["changed"] is True
    assert action_event["shape"] == [40]
    assert action_event["input_sha256"] != action_event["output_sha256"]


def test_runtime_applies_locked_root_velocity_when_ball_moves(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    env = _Env(torch)
    env.scene["object"].data.root_vel_w[0, 0] = 0.3
    runtime = HumanoidArenaRecoveryRuntime(
        SUITE,
        "support-state-push",
        control_dt_s=0.02,
        simulator_revision="isaac-test",
        output_dir=tmp_path,
    )
    runtime.reset(env, episode_seed=9)
    runtime.before_control_step(env, task_success=False)
    assert runtime.triggered
    velocity = env.scene["robot"].data.root_vel_w[0, :3]
    torch.testing.assert_close(velocity, torch.tensor([0.0, 0.45, 0.0]))
