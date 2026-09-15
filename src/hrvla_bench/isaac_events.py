"""Deterministic Isaac Lab failure injectors used by benchmark pilots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _append_audit(path_value: str | None, record: dict[str, Any]) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def reset_one_shot_injectors(env: Any, env_ids: Any = None) -> None:
    """Reset HRVLA one-shot flags explicitly at an episode boundary."""

    import torch

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    registry = getattr(env, "_hrvla_one_shot_registry", None)
    if registry is None:
        env._hrvla_one_shot_registry = {}
        return
    for flags in registry.values():
        flags[env_ids] = False


def _pending_one_shot_envs(env: Any, env_ids: Any, injector_id: str, device: Any):
    import torch

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device)
    registry = getattr(env, "_hrvla_one_shot_registry", None)
    if registry is None:
        registry = {}
        env._hrvla_one_shot_registry = registry
    flags = registry.get(injector_id)
    if flags is None:
        flags = torch.zeros(env.scene.num_envs, dtype=torch.bool, device=device)
        registry[injector_id] = flags
    return env_ids[~flags[env_ids]], flags


def apply_root_velocity_delta_once(
    env: Any,
    env_ids: Any,
    *,
    injector_id: str,
    linear_velocity_delta: tuple[float, float, float],
    angular_velocity_delta: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_name: str = "robot",
    audit_path: str | None = None,
    episode_seed: int | None = None,
) -> None:
    """Apply one deterministic world-frame velocity delta per reset episode."""

    import torch

    asset = env.scene[asset_name]
    pending, flags = _pending_one_shot_envs(env, env_ids, injector_id, asset.device)
    if pending.numel() == 0:
        return
    delta = torch.tensor(
        [*linear_velocity_delta, *angular_velocity_delta],
        dtype=asset.data.root_vel_w.dtype,
        device=asset.device,
    )
    before = asset.data.root_vel_w[pending].clone()
    after = before + delta.unsqueeze(0)
    asset.write_root_velocity_to_sim(after, env_ids=pending)
    flags[pending] = True
    _append_audit(
        audit_path,
        {
            "event": "apply_root_velocity_delta_once",
            "injector_id": injector_id,
            "asset_name": asset_name,
            "environment_ids": [int(value) for value in pending.detach().cpu().tolist()],
            "episode_seed": None if episode_seed is None else int(episode_seed),
            "episode_steps": [
                int(value) for value in env.episode_length_buf[pending].detach().cpu().tolist()
            ],
            "linear_velocity_delta": [float(value) for value in linear_velocity_delta],
            "angular_velocity_delta": [float(value) for value in angular_velocity_delta],
            "velocity_before": before.detach().cpu().tolist(),
            "velocity_after": after.detach().cpu().tolist(),
        },
    )


def apply_action_window(
    action: Any,
    *,
    injector_id: str,
    elapsed_s: float,
    parameters: dict[str, Any],
) -> Any:
    """Apply a deterministic action-stream perturbation and return a copy."""

    import numpy as np

    values = np.asarray(action, dtype=np.float32)
    if values.ndim not in (1, 2):
        raise ValueError(f"action must be a vector or chunk, got {values.shape}")
    output = values.copy()
    duration_s = float(parameters.get("duration_s", 0.0))
    if duration_s <= 0.0:
        raise ValueError("action injector duration_s must be positive")
    if elapsed_s < 0.0:
        raise ValueError("elapsed_s must be non-negative")
    if elapsed_s >= duration_s:
        return output

    if injector_id == "release-grasp-contact":
        if output.shape[-1] != 40:
            raise ValueError("release-grasp-contact requires semantic action40")
        output[..., 38:40] = 0.0
    elif injector_id == "attenuate-sonic-latent":
        if output.shape[-1] != 64:
            raise ValueError("attenuate-sonic-latent requires the post-encoder latent64")
        scale = float(parameters.get("scale", -1.0))
        if not 0.0 <= scale < 1.0:
            raise ValueError("latent attenuation scale must be in [0, 1)")
        output *= scale
    else:
        raise ValueError(f"unsupported action injector: {injector_id}")
    return output


def push_once_by_setting_velocity(
    env: Any,
    env_ids: Any,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: Any = None,
    audit_path: str | None = None,
) -> None:
    """Apply Isaac Lab's root-velocity push at most once to each environment.

    The event manager may schedule an interval term repeatedly. This wrapper
    preserves the first semantic injection and suppresses later calls so a
    single-failure episode cannot silently become a periodic-disturbance test.
    """
    from isaaclab.envs.mdp import push_by_setting_velocity
    from isaaclab.managers import SceneEntityCfg

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    device = env.scene[asset_cfg.name].device
    pending, flags = _pending_one_shot_envs(
        env, env_ids, "push_once_by_setting_velocity", device
    )
    if pending.numel() == 0:
        return
    push_by_setting_velocity(env, pending, velocity_range, asset_cfg)
    flags[pending] = True
    if audit_path:
        record = {
            "event": "push_once_by_setting_velocity",
            "environment_ids": [int(value) for value in pending.detach().cpu().tolist()],
            "episode_steps": [
                int(value)
                for value in env.episode_length_buf[pending].detach().cpu().tolist()
            ],
            "velocity_range": {
                key: [float(bounds[0]), float(bounds[1])]
                for key, bounds in sorted(velocity_range.items())
            },
        }
        _append_audit(audit_path, record)
        print(
            f"[HRVLA] audited one-shot root-velocity push for "
            f"{len(record['environment_ids'])} environment(s): {audit_path}"
        )
