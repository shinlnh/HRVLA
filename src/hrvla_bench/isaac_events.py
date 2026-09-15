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

    clear_expired_body_impulses(env, force=True)
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


def _quat_apply_wxyz(quaternion: Any, vector: Any) -> Any:
    """Rotate batched 3-D vectors without importing Isaac Lab utilities."""

    import torch

    scalar = quaternion[..., :1]
    xyz = quaternion[..., 1:]
    twice_cross = 2.0 * torch.linalg.cross(xyz, vector, dim=-1)
    return vector + scalar * twice_cross + torch.linalg.cross(xyz, twice_cross, dim=-1)


def _root_link_pose(asset: Any, env_ids: Any) -> Any:
    pose = getattr(asset.data, "root_link_pose_w", None)
    if pose is None:
        state = getattr(asset.data, "root_state_w", None)
        if state is None:
            raise ValueError("asset does not expose root_link_pose_w or root_state_w")
        pose = state[..., :7]
    return pose[env_ids].clone()


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


def apply_root_local_lateral_velocity_once(
    env: Any,
    env_ids: Any,
    *,
    injector_id: str,
    lateral_mps: float,
    lateral_direction_robot: str,
    asset_name: str = "robot",
    audit_path: str | None = None,
    episode_seed: int | None = None,
) -> None:
    """Apply a world velocity delta resolved from the asset's local lateral axis."""

    import torch

    if lateral_mps <= 0.0:
        raise ValueError("lateral_mps must be positive")
    if lateral_direction_robot not in {"left", "right"}:
        raise ValueError("lateral_direction_robot must be 'left' or 'right'")
    asset = env.scene[asset_name]
    pending, flags = _pending_one_shot_envs(env, env_ids, injector_id, asset.device)
    if pending.numel() == 0:
        return
    pose = _root_link_pose(asset, pending)
    sign = 1.0 if lateral_direction_robot == "left" else -1.0
    local = torch.zeros((len(pending), 3), dtype=pose.dtype, device=asset.device)
    local[:, 1] = sign * float(lateral_mps)
    world = _quat_apply_wxyz(pose[:, 3:7], local)
    world[:, 2] = 0.0
    before = asset.data.root_vel_w[pending].clone()
    after = before.clone()
    after[:, :3] += world
    asset.write_root_velocity_to_sim(after, env_ids=pending)
    flags[pending] = True
    _append_audit(
        audit_path,
        {
            "event": "apply_root_local_lateral_velocity_once",
            "injector_id": injector_id,
            "asset_name": asset_name,
            "environment_ids": [int(value) for value in pending.detach().cpu().tolist()],
            "episode_seed": None if episode_seed is None else int(episode_seed),
            "episode_steps": [
                int(value) for value in env.episode_length_buf[pending].detach().cpu().tolist()
            ],
            "lateral_mps": float(lateral_mps),
            "lateral_direction_robot": lateral_direction_robot,
            "world_velocity_delta": world.detach().cpu().tolist(),
            "velocity_before": before.detach().cpu().tolist(),
            "velocity_after": after.detach().cpu().tolist(),
        },
    )


def apply_asset_local_translation_once(
    env: Any,
    env_ids: Any,
    *,
    injector_id: str,
    asset_name: str,
    local_translation_m: tuple[float, float, float],
    audit_path: str | None = None,
    episode_seed: int | None = None,
) -> None:
    """Translate an asset once along its current local frame, preserving orientation."""

    import torch

    asset = env.scene[asset_name]
    pending, flags = _pending_one_shot_envs(env, env_ids, injector_id, asset.device)
    if pending.numel() == 0:
        return
    before = _root_link_pose(asset, pending)
    local = torch.tensor(local_translation_m, dtype=before.dtype, device=asset.device)
    local = local.unsqueeze(0).repeat(len(pending), 1)
    world_delta = _quat_apply_wxyz(before[:, 3:7], local)
    after = before.clone()
    after[:, :3] += world_delta
    asset.write_root_pose_to_sim(after, env_ids=pending)
    flags[pending] = True
    _append_audit(
        audit_path,
        {
            "event": "apply_asset_local_translation_once",
            "injector_id": injector_id,
            "asset_name": asset_name,
            "environment_ids": [int(value) for value in pending.detach().cpu().tolist()],
            "episode_seed": None if episode_seed is None else int(episode_seed),
            "episode_steps": [
                int(value) for value in env.episode_length_buf[pending].detach().cpu().tolist()
            ],
            "local_translation_m": [float(value) for value in local_translation_m],
            "pose_before": before.detach().cpu().tolist(),
            "pose_after": after.detach().cpu().tolist(),
        },
    )


def place_asset_relative_once(
    env: Any,
    env_ids: Any,
    *,
    injector_id: str,
    asset_name: str,
    reference_asset_name: str,
    reference_local_position_m: tuple[float, float, float],
    align_orientation: bool = True,
    audit_path: str | None = None,
    episode_seed: int | None = None,
) -> None:
    """Place one movable asset at a deterministic pose relative to another asset."""

    import torch

    asset = env.scene[asset_name]
    reference = env.scene[reference_asset_name]
    pending, flags = _pending_one_shot_envs(env, env_ids, injector_id, asset.device)
    if pending.numel() == 0:
        return
    before = _root_link_pose(asset, pending)
    reference_pose = _root_link_pose(reference, pending)
    local = torch.tensor(
        reference_local_position_m, dtype=before.dtype, device=asset.device
    ).unsqueeze(0).repeat(len(pending), 1)
    after = before.clone()
    after[:, :3] = reference_pose[:, :3] + _quat_apply_wxyz(
        reference_pose[:, 3:7], local
    )
    if align_orientation:
        after[:, 3:7] = reference_pose[:, 3:7]
    asset.write_root_pose_to_sim(after, env_ids=pending)
    zero_velocity = torch.zeros((len(pending), 6), dtype=before.dtype, device=asset.device)
    asset.write_root_velocity_to_sim(zero_velocity, env_ids=pending)
    flags[pending] = True
    _append_audit(
        audit_path,
        {
            "event": "place_asset_relative_once",
            "injector_id": injector_id,
            "asset_name": asset_name,
            "reference_asset_name": reference_asset_name,
            "environment_ids": [int(value) for value in pending.detach().cpu().tolist()],
            "episode_seed": None if episode_seed is None else int(episode_seed),
            "episode_steps": [
                int(value) for value in env.episode_length_buf[pending].detach().cpu().tolist()
            ],
            "reference_local_position_m": [
                float(value) for value in reference_local_position_m
            ],
            "align_orientation": bool(align_orientation),
            "pose_before": before.detach().cpu().tolist(),
            "pose_after": after.detach().cpu().tolist(),
        },
    )


def apply_body_impulse_once(
    env: Any,
    env_ids: Any,
    *,
    injector_id: str,
    asset_name: str,
    body_name: str,
    impulse_world_ns: tuple[float, float, float],
    control_dt_s: float,
    audit_path: str | None = None,
    episode_seed: int | None = None,
) -> None:
    """Apply an exact world-frame impulse as a one-control-step external force."""

    import torch

    if control_dt_s <= 0.0:
        raise ValueError("control_dt_s must be positive")
    asset = env.scene[asset_name]
    body_names = list(getattr(asset, "body_names", None) or asset.data.body_names)
    matches = [index for index, name in enumerate(body_names) if name == body_name]
    if len(matches) != 1:
        raise ValueError(f"body {body_name!r} did not resolve exactly once in {asset_name}")
    pending, flags = _pending_one_shot_envs(env, env_ids, injector_id, asset.device)
    if pending.numel() == 0:
        return
    impulse = torch.tensor(impulse_world_ns, dtype=torch.float32, device=asset.device)
    if not bool(torch.isfinite(impulse).all()) or float(torch.linalg.vector_norm(impulse)) <= 0.0:
        raise ValueError("impulse_world_ns must be finite and non-zero")
    force = (impulse / float(control_dt_s)).reshape(1, 1, 3).repeat(len(pending), 1, 1)
    torque = torch.zeros_like(force)
    body_ids = torch.tensor(matches, dtype=torch.long, device=asset.device)
    asset.set_external_force_and_torque(
        force,
        torque,
        body_ids=body_ids,
        env_ids=pending,
        is_global=True,
    )
    current_steps = env.episode_length_buf[pending].detach().clone()
    active = getattr(env, "_hrvla_active_body_impulses", None)
    if active is None:
        active = []
        env._hrvla_active_body_impulses = active
    active.append(
        {
            "asset_name": asset_name,
            "body_ids": body_ids,
            "env_ids": pending.detach().clone(),
            "clear_after_steps": current_steps + 1,
        }
    )
    flags[pending] = True
    _append_audit(
        audit_path,
        {
            "event": "apply_body_impulse_once",
            "injector_id": injector_id,
            "asset_name": asset_name,
            "body_name": body_name,
            "environment_ids": [int(value) for value in pending.detach().cpu().tolist()],
            "episode_seed": None if episode_seed is None else int(episode_seed),
            "episode_steps": [int(value) for value in current_steps.cpu().tolist()],
            "impulse_world_ns": [float(value) for value in impulse_world_ns],
            "control_dt_s": float(control_dt_s),
            "force_world_n": force[0, 0].detach().cpu().tolist(),
        },
    )


def clear_expired_body_impulses(env: Any, *, force: bool = False) -> int:
    """Clear body-force buffers after exactly one control step or on reset."""

    import torch

    active = list(getattr(env, "_hrvla_active_body_impulses", []) or [])
    remaining = []
    cleared = 0
    for record in active:
        env_ids = record["env_ids"]
        expired = force or bool(
            torch.all(env.episode_length_buf[env_ids] >= record["clear_after_steps"])
        )
        if not expired:
            remaining.append(record)
            continue
        asset = env.scene[record["asset_name"]]
        zeros = torch.zeros(
            (len(env_ids), len(record["body_ids"]), 3),
            dtype=torch.float32,
            device=asset.device,
        )
        asset.set_external_force_and_torque(
            zeros,
            zeros,
            body_ids=record["body_ids"],
            env_ids=env_ids,
            is_global=True,
        )
        cleared += 1
    env._hrvla_active_body_impulses = remaining
    return cleared
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
