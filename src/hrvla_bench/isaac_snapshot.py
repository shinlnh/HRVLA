"""Deterministic, hash-addressed Isaac Lab scene snapshots for recovery cells."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _values(tensor: Any) -> list[Any]:
    value = tensor
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _snapshot_hash(core: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(core)).hexdigest()


def capture_scene_snapshot(
    env: Any,
    *,
    task_id: str,
    event_id: str,
    episode_seed: int,
    scene_assets: Iterable[str],
    simulator_revision: str,
    env_id: int = 0,
) -> dict[str, Any]:
    """Capture root and optional articulation joint state for one environment."""

    asset_names = tuple(sorted(set(str(name) for name in scene_assets)))
    if not asset_names:
        raise ValueError("a recovery snapshot must include at least one scene asset")
    if not task_id or not event_id or not simulator_revision:
        raise ValueError("task, event, and simulator revision are required")
    if env_id < 0 or env_id >= int(env.num_envs):
        raise ValueError(f"environment index {env_id} is out of range")

    assets: dict[str, Any] = {}
    for name in asset_names:
        asset = env.scene[name]
        root_state = _values(asset.data.root_state_w[env_id])
        if len(root_state) != 13:
            raise ValueError(f"asset {name} root state has width {len(root_state)}, expected 13")
        record: dict[str, Any] = {
            "asset_type": type(asset).__name__,
            "root_state_w": root_state,
        }
        joint_pos = getattr(asset.data, "joint_pos", None)
        joint_vel = getattr(asset.data, "joint_vel", None)
        if (joint_pos is None) != (joint_vel is None):
            raise ValueError(f"asset {name} exposes only half of its joint state")
        if joint_pos is not None:
            positions = _values(joint_pos[env_id])
            velocities = _values(joint_vel[env_id])
            if len(positions) != len(velocities):
                raise ValueError(f"asset {name} joint position/velocity widths differ")
            joint_names = [str(value) for value in getattr(asset, "joint_names", [])]
            if joint_names and len(joint_names) != len(positions):
                raise ValueError(f"asset {name} joint names/state widths differ")
            record.update(
                {
                    "joint_names": joint_names,
                    "joint_pos": positions,
                    "joint_vel": velocities,
                }
            )
        assets[name] = record

    episode_step = int(_values(env.episode_length_buf[env_id : env_id + 1])[0])
    core = {
        "schema_version": 1,
        "task_id": task_id,
        "event_id": event_id,
        "episode_seed": int(episode_seed),
        "episode_step": episode_step,
        "environment_index": int(env_id),
        "simulator_revision": simulator_revision,
        "assets": assets,
    }
    return {**core, "snapshot_sha256": _snapshot_hash(core)}


def validate_snapshot(snapshot: dict[str, Any]) -> str:
    claimed = snapshot.get("snapshot_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError("snapshot_sha256 is missing or malformed")
    core = {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    actual = _snapshot_hash(core)
    if actual != claimed:
        raise ValueError(f"snapshot hash mismatch: {actual} != {claimed}")
    if snapshot.get("schema_version") != 1:
        raise ValueError("unsupported snapshot schema")
    for name, asset in snapshot.get("assets", {}).items():
        if len(asset.get("root_state_w", [])) != 13:
            raise ValueError(f"asset {name} has a malformed root state")
        if ("joint_pos" in asset) != ("joint_vel" in asset):
            raise ValueError(f"asset {name} has an incomplete joint state")
        if "joint_pos" in asset and len(asset["joint_pos"]) != len(asset["joint_vel"]):
            raise ValueError(f"asset {name} joint widths differ")
    return claimed


def write_snapshot_atomic(path: Path, snapshot: dict[str, Any]) -> None:
    """Write only a validated immutable snapshot, without wall-clock metadata."""

    validate_snapshot(snapshot)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_bytes(snapshot) + b"\n")
    os.replace(temporary, path)


def load_snapshot(path: Path) -> dict[str, Any]:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    validate_snapshot(snapshot)
    return snapshot


def snapshot_state_sha256(snapshot: dict[str, Any]) -> str:
    """Hash only restorable scene state and joint layout, excluding provenance."""

    validate_snapshot(snapshot)
    return _snapshot_hash({"assets": snapshot["assets"]})


def restore_scene_snapshot(env: Any, snapshot: dict[str, Any]) -> None:
    """Restore a validated snapshot through Isaac Lab's public state writers."""

    validate_snapshot(snapshot)
    import torch

    env_id = int(snapshot["environment_index"])
    for name, record in snapshot["assets"].items():
        asset = env.scene[name]
        device = asset.device
        env_ids = torch.tensor([env_id], dtype=torch.long, device=device)
        root_state = torch.as_tensor(
            record["root_state_w"], dtype=asset.data.root_state_w.dtype, device=device
        ).reshape(1, 13)
        asset.write_root_state_to_sim(root_state, env_ids=env_ids)
        if "joint_pos" in record:
            actual_names = [str(value) for value in getattr(asset, "joint_names", [])]
            if record["joint_names"] and actual_names != record["joint_names"]:
                raise ValueError(f"asset {name} joint ordering changed before restore")
            position = torch.as_tensor(
                record["joint_pos"], dtype=asset.data.joint_pos.dtype, device=device
            ).reshape(1, -1)
            velocity = torch.as_tensor(
                record["joint_vel"], dtype=asset.data.joint_vel.dtype, device=device
            ).reshape(1, -1)
            asset.write_joint_state_to_sim(position, velocity, env_ids=env_ids)
    env.scene.write_data_to_sim()
    simulation = getattr(env, "sim", None)
    forward = getattr(simulation, "forward", None)
    if callable(forward):
        forward()
    has_rtx_sensors = getattr(simulation, "has_rtx_sensors", None)
    rerender = bool(getattr(getattr(env, "cfg", None), "rerender_on_reset", False))
    render = getattr(simulation, "render", None)
    if callable(has_rtx_sensors) and has_rtx_sensors() and rerender and callable(render):
        render()
