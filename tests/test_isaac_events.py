from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from hrvla_bench.isaac_events import (
    apply_action_window,
    apply_root_velocity_delta_once,
    push_once_by_setting_velocity,
    reset_one_shot_injectors,
)


class _FakeAsset:
    def __init__(self, num_envs: int, torch_module) -> None:
        self.device = torch_module.device("cpu")
        self.data = types.SimpleNamespace(root_vel_w=torch_module.zeros((num_envs, 6)))
        self.writes = []

    def write_root_velocity_to_sim(self, values, *, env_ids) -> None:
        self.data.root_vel_w[env_ids] = values
        self.writes.append((values.clone(), env_ids.clone()))


class _FakeScene(dict):
    def __init__(self, num_envs: int, torch_module) -> None:
        super().__init__(robot=_FakeAsset(num_envs, torch_module))
        self.num_envs = num_envs


class _FakeEnv:
    def __init__(self, torch_module, num_envs: int = 3) -> None:
        self.device = torch_module.device("cpu")
        self.scene = _FakeScene(num_envs, torch_module)
        self.episode_length_buf = torch_module.arange(num_envs)


def test_release_grasp_opens_only_binary_hands_inside_window() -> None:
    action = np.arange(40, dtype=np.float32)
    released = apply_action_window(
        action,
        injector_id="release-grasp-contact",
        elapsed_s=0.1,
        parameters={"duration_s": 0.2},
    )
    np.testing.assert_array_equal(released[:38], action[:38])
    np.testing.assert_array_equal(released[38:], [0.0, 0.0])
    np.testing.assert_array_equal(action, np.arange(40, dtype=np.float32))
    after = apply_action_window(
        action,
        injector_id="release-grasp-contact",
        elapsed_s=0.2,
        parameters={"duration_s": 0.2},
    )
    np.testing.assert_array_equal(after, action)


def test_tracker_underexecution_scales_only_post_encoder_latent() -> None:
    latent = np.ones((3, 64), dtype=np.float32)
    scaled = apply_action_window(
        latent,
        injector_id="attenuate-sonic-latent",
        elapsed_s=0.3,
        parameters={"duration_s": 0.6, "scale": 0.55},
    )
    np.testing.assert_allclose(scaled, 0.55)
    with pytest.raises(ValueError, match="latent64"):
        apply_action_window(
            np.ones(40),
            injector_id="attenuate-sonic-latent",
            elapsed_s=0.3,
            parameters={"duration_s": 0.6, "scale": 0.55},
        )


def test_action_injector_rejects_unlocked_parameters() -> None:
    with pytest.raises(ValueError, match="positive"):
        apply_action_window(
            np.ones(40),
            injector_id="release-grasp-contact",
            elapsed_s=0.0,
            parameters={"duration_s": 0.0},
        )
    with pytest.raises(ValueError, match="unsupported"):
        apply_action_window(
            np.ones(40),
            injector_id="invented",
            elapsed_s=0.0,
            parameters={"duration_s": 1.0},
        )


def test_root_velocity_delta_is_one_shot_until_explicit_reset(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    env = _FakeEnv(torch)
    audit_path = tmp_path / "velocity.jsonl"
    selected = torch.tensor([0, 2])
    kwargs = {
        "injector_id": "support-foot-slip",
        "linear_velocity_delta": (0.0, 0.35, 0.0),
        "audit_path": str(audit_path),
        "episode_seed": 17,
    }
    apply_root_velocity_delta_once(env, selected, **kwargs)
    apply_root_velocity_delta_once(env, selected, **kwargs)
    assert len(env.scene["robot"].writes) == 1
    torch.testing.assert_close(
        env.scene["robot"].data.root_vel_w[selected, 1], torch.tensor([0.35, 0.35])
    )

    reset_one_shot_injectors(env, torch.tensor([2]))
    apply_root_velocity_delta_once(env, selected, **kwargs)
    assert len(env.scene["robot"].writes) == 2
    torch.testing.assert_close(
        env.scene["robot"].data.root_vel_w[:, 1], torch.tensor([0.35, 0.0, 0.70])
    )
    records = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert [record["environment_ids"] for record in records] == [[0, 2], [2]]
    assert all(record["episode_seed"] == 17 for record in records)


def test_legacy_push_wrapper_uses_resettable_registry(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    calls = []

    class SceneEntityCfg:
        def __init__(self, name: str) -> None:
            self.name = name

    def fake_push(env, env_ids, velocity_range, asset_cfg) -> None:
        calls.append(env_ids.clone())

    isaaclab = types.ModuleType("isaaclab")
    envs = types.ModuleType("isaaclab.envs")
    mdp = types.ModuleType("isaaclab.envs.mdp")
    managers = types.ModuleType("isaaclab.managers")
    mdp.push_by_setting_velocity = fake_push
    managers.SceneEntityCfg = SceneEntityCfg
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.envs", envs)
    monkeypatch.setitem(sys.modules, "isaaclab.envs.mdp", mdp)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)

    env = _FakeEnv(torch, num_envs=2)
    velocity_range = {"x": (0.1, 0.1)}
    push_once_by_setting_velocity(env, None, velocity_range)
    push_once_by_setting_velocity(env, None, velocity_range)
    reset_one_shot_injectors(env)
    push_once_by_setting_velocity(env, None, velocity_range)
    assert len(calls) == 2
    torch.testing.assert_close(calls[0], torch.tensor([0, 1]))
    torch.testing.assert_close(calls[1], torch.tensor([0, 1]))
