"""Deterministic Isaac Lab failure injectors used by benchmark pilots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
    import torch
    from isaaclab.envs.mdp import push_by_setting_velocity
    from isaaclab.managers import SceneEntityCfg

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    device = env.scene[asset_cfg.name].device
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device)
    flags = getattr(env, "_hrvla_push_once_applied", None)
    if flags is None:
        flags = torch.zeros(env.scene.num_envs, dtype=torch.bool, device=device)
        env._hrvla_push_once_applied = flags
    pending = env_ids[~flags[env_ids]]
    if pending.numel() == 0:
        return
    push_by_setting_velocity(env, pending, velocity_range, asset_cfg)
    flags[pending] = True
    if audit_path:
        path = Path(audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
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
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            f"[HRVLA] audited one-shot root-velocity push for "
            f"{len(record['environment_ids'])} environment(s): {path}"
        )
