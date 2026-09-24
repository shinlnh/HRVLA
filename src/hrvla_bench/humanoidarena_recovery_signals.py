"""Live HumanoidArena geometry signals used by recovery event detectors."""

from __future__ import annotations

import importlib
import math
from typing import Any, Mapping


_WRIST_BODY_TOKENS = (
    "hand_palm",
    "hand_base",
    "hand_camera_base",
    "wrist_yaw",
    "wrist_pitch",
    "wrist_roll",
)


def _body_names(asset: Any) -> list[str]:
    data = getattr(asset, "data", None)
    names = getattr(asset, "body_names", None) or getattr(data, "body_names", None)
    if not names:
        raise ValueError("asset does not expose body_names")
    return [str(name) for name in names]


def _body_positions(asset: Any) -> Any:
    data = getattr(asset, "data", None)
    positions = getattr(data, "body_link_pos_w", None)
    if positions is None:
        positions = getattr(data, "body_pos_w", None)
    if positions is None:
        state = getattr(data, "body_state_w", None)
        if state is not None:
            positions = state[..., :3]
    if positions is None:
        raise ValueError("asset does not expose world-frame body positions")
    return positions


def _usd_door_handle_position(cfg: Any, stage: Any, env_id: int, cache: Any) -> tuple[str, tuple[float, float, float]]:
    """Use the pinned OpenDoor task's handle prim when its scene asset is XFormPrim."""

    resolver = getattr(cfg, "_get_open_door_prims", None)
    if not callable(resolver):
        raise ValueError("OpenDoor task does not expose its pinned USD handle resolver")
    handle = resolver(stage, env_id).get("handle")
    if handle is None or not handle.IsValid():
        raise ValueError("OpenDoor USD handle prim is unavailable")
    translation = cache.GetLocalToWorldTransform(handle).ExtractTranslation()
    return str(handle.GetName()), tuple(float(value) for value in translation)


def _scalar(value: Any, env_id: int = 0) -> float:
    item = value[env_id]
    if hasattr(item, "detach"):
        item = item.detach()
    if hasattr(item, "cpu"):
        item = item.cpu()
    if hasattr(item, "item"):
        item = item.item()
    return float(item)


def _boolean(value: Any, env_id: int = 0) -> bool:
    item = value[env_id]
    if hasattr(item, "detach"):
        item = item.detach()
    if hasattr(item, "cpu"):
        item = item.cpu()
    if hasattr(item, "item"):
        item = item.item()
    return bool(item)


def _open_door_signals(env: Any, door_asset_name: str) -> dict[str, Any]:
    import torch

    cfg = getattr(env, "cfg", None)
    diagnostics_fn = getattr(cfg, "get_open_door_reward_diagnostics", None)
    if not callable(diagnostics_fn):
        raise ValueError("open-door runtime diagnostics are unavailable")
    diagnostics = diagnostics_fn(env)
    output = {
        "door_leaf_angle_deg": _scalar(diagnostics["leaf_angle_deg"]),
        "door_latch_unlocked": _boolean(diagnostics["latch_unlocked"]),
    }

    door = env.scene[door_asset_name]
    robot = env.scene["robot"]
    robot_names = _body_names(robot)
    wrist_ids = [
        index
        for index, name in enumerate(robot_names)
        if any(token in name.lower() for token in _WRIST_BODY_TOKENS)
    ]
    if not wrist_ids:
        raise ValueError(f"no wrist proxy found in robot body_names: {robot_names[:20]}")
    wrist_positions = _body_positions(robot)[0, wrist_ids, :3]
    if getattr(door, "data", None) is not None:
        door_names = _body_names(door)
        handle_ids = [index for index, name in enumerate(door_names) if "handle" in name.lower()]
        if not handle_ids:
            raise ValueError(f"no handle body found in door body_names: {door_names}")
        handle_positions = _body_positions(door)[0, handle_ids, :3]
        handle_body_names = [door_names[index] for index in handle_ids]
    else:
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        handle_name, handle_position = _usd_door_handle_position(cfg, stage, 0, cache)
        handle_positions = torch.as_tensor(
            handle_position, device=wrist_positions.device, dtype=wrist_positions.dtype
        ).reshape(1, 3)
        handle_body_names = [handle_name]
    deltas = wrist_positions[:, None, :] - handle_positions[None, :, :]
    output["wrist_handle_min_distance_m"] = float(
        torch.linalg.vector_norm(deltas, dim=-1).min().detach().cpu().item()
    )
    output["door_handle_body_names"] = handle_body_names
    output["wrist_body_names"] = [robot_names[index] for index in wrist_ids]
    return output


def _root_upright(
    env: Any, *, minimum_root_height_m: float, minimum_up_axis_z: float
) -> tuple[bool, float, float]:
    state = env.scene["robot"].data.root_state_w[0]
    values = state.detach().cpu().tolist() if hasattr(state, "detach") else state.tolist()
    height = float(values[2])
    w, x, y, _z = (float(value) for value in values[3:7])
    norm = math.sqrt(sum(float(value) ** 2 for value in values[3:7]))
    if norm <= 1e-8:
        raise ValueError("robot root quaternion is degenerate")
    x /= norm
    y /= norm
    up_axis_z = 1.0 - 2.0 * (x * x + y * y)
    return (
        height >= minimum_root_height_m and up_axis_z >= minimum_up_axis_z,
        height,
        up_axis_z,
    )


def _point_box_xy_distance(point: Any, box: tuple[float, ...]) -> float:
    x, y = float(point[0]), float(point[1])
    x_lo, x_hi, y_lo, y_hi = (float(value) for value in box[:4])
    dx = max(x_lo - x, 0.0, x - x_hi)
    dy = max(y_lo - y, 0.0, y - y_hi)
    return math.hypot(dx, dy)


def _seat_signals(env: Any, parameters: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    rewards = importlib.import_module(
        "tasks.g1_tasks.move_sit_sofa_g1_29dof_dex3_wholebody.mdp.rewards"
    )
    boxes = rewards._get_sofa_seat_boxes_world(env)
    box = boxes[0]
    if box is None:
        raise ValueError("live sofa-seat AABB is unavailable")
    body_positions = rewards._get_body_positions_w(env)
    body_ids = rewards._get_sit_body_indices(env)
    positions = body_positions[0, list(body_ids), :2]
    if hasattr(positions, "detach"):
        positions = positions.detach().cpu()
    distances = [_point_box_xy_distance(point, box) for point in positions.tolist()]
    upright, height, up_axis_z = _root_upright(
        env,
        minimum_root_height_m=float(parameters["minimum_root_height_m"]),
        minimum_up_axis_z=float(parameters["minimum_up_axis_z"]),
    )
    robot_names = _body_names(env.scene["robot"])
    ankle_ids = [
        index
        for index, name in enumerate(robot_names)
        if name in {"left_ankle_roll_link", "right_ankle_roll_link"}
    ]
    if len(ankle_ids) != 2:
        raise ValueError("left/right ankle-roll bodies are required for support-foot selection")
    contact = rewards._get_body_contact_forces_w(env)[0, ankle_ids, :]
    support_offset = int(torch.linalg.vector_norm(contact, dim=-1).argmax().item())
    support_body_name = robot_names[ankle_ids[support_offset]]
    return {
        "seat_xy_distance_m": min(distances),
        "robot_upright": upright,
        "root_height_m": height,
        "root_up_axis_z": up_axis_z,
        "seat_proxy_body_names": list(getattr(env, "_sit_sofa_body_names", ())),
        "support_foot_body_name": support_body_name,
        "support_foot_contact_force_n": float(
            torch.linalg.vector_norm(contact[support_offset]).detach().cpu().item()
        ),
    }


def _boxing_signals(env: Any) -> dict[str, Any]:
    import torch

    rewards = importlib.import_module(
        "tasks.g1_tasks.move_boxing_bag_g1_29dof_dex3_wholebody.mdp.rewards"
    )
    body_positions = rewards._get_body_positions_w(env)
    body_ids = rewards._get_punch_body_indices(env)
    punch_positions = body_positions[:, list(body_ids), :]
    target_positions = rewards._get_boxing_target_positions_world(env)
    thresholds = rewards._get_boxing_target_hit_distance_thresholds(env)
    distances = torch.linalg.vector_norm(
        punch_positions - target_positions.unsqueeze(1), dim=-1
    )
    clearance = distances - thresholds.unsqueeze(1)
    flat_index = int(clearance[0].argmin().detach().cpu().item())
    nearest_punch = punch_positions[0, flat_index]
    away = nearest_punch - target_positions[0]
    away_norm = torch.linalg.vector_norm(away)
    if float(away_norm.detach().cpu().item()) <= 1e-8:
        raise ValueError("boxing recoil direction is degenerate at the target center")
    recoil_direction = away / away_norm
    return {
        "punch_hit_clearance_m": float(clearance[0].min().detach().cpu().item()),
        "punch_body_names": list(getattr(env, "_boxing_bag_punch_body_names", ())),
        "recoil_direction_world": recoil_direction.detach().cpu().tolist(),
    }


def signals_for_detector(
    detector_id: str,
    env: Any,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve only the source-backed live signals needed by one detector."""

    if detector_id in {"wrist-handle-approach", "door-open-before-traversal"}:
        return _open_door_signals(
            env, str(parameters.get("door_asset_name", "door"))
        )
    if detector_id == "seat-approach":
        return _seat_signals(env, parameters)
    if detector_id == "strike-approach-shell":
        return _boxing_signals(env)
    return {}


__all__ = ["signals_for_detector"]
