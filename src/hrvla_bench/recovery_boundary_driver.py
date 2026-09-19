"""Pre-registered deterministic drivers that materialize recovery boundaries.

These fixtures establish a semantic *pre-failure* condition when the released
reference policy cannot reliably reach it.  They stop at the detector edge and
never provide a post-failure recovery action.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .plan import canonical_sha256


_WRIST_BODY_TOKENS = (
    "hand_palm",
    "hand_base",
    "hand_camera_base",
    "wrist_yaw",
    "wrist_pitch",
    "wrist_roll",
)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _body_names(asset: Any) -> list[str]:
    names = getattr(asset, "body_names", None) or getattr(asset.data, "body_names", None)
    if not names:
        raise ValueError("boundary driver asset does not expose body_names")
    return [str(name) for name in names]


def _body_positions(asset: Any) -> Any:
    positions = getattr(asset.data, "body_link_pos_w", None)
    if positions is None:
        positions = getattr(asset.data, "body_pos_w", None)
    if positions is None:
        state = getattr(asset.data, "body_state_w", None)
        if state is not None:
            positions = state[..., :3]
    if positions is None:
        raise ValueError("boundary driver asset does not expose body positions")
    return positions


def _flush(env: Any) -> None:
    write = getattr(env.scene, "write_data_to_sim", None)
    if callable(write):
        write()
    forward = getattr(getattr(env, "sim", None), "forward", None)
    if callable(forward):
        forward()


def _write_root_state(asset: Any, state: Any, env_ids: Any) -> None:
    writer = getattr(asset, "write_root_state_to_sim", None)
    if callable(writer):
        writer(state, env_ids=env_ids)
        return
    asset.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(state[:, 7:13], env_ids=env_ids)


class DeterministicBoundaryDriver:
    """Apply one suite-hashed reachability fixture until the detector fires."""

    def __init__(
        self,
        scenario: dict[str, Any],
        *,
        trace_path: Path,
        episode_seed: int,
        env_id: int = 0,
    ) -> None:
        self.scenario_id = str(scenario["id"])
        self.spec = scenario.get("boundary_driver")
        self.trace_path = Path(trace_path)
        self.episode_seed = int(episode_seed)
        self.env_id = int(env_id)
        self.active = self.spec is not None
        self._target_root_state = None
        self._target_asset_name: str | None = None
        self._target_root_schedule: list[Any] = []
        self._target_root_schedule_index = 0
        self._target_joint_asset_name: str | None = None
        self._target_joint_positions = None
        self._target_joint_velocities = None
        self._action_samples = 0

    @property
    def driver_id(self) -> str | None:
        return None if self.spec is None else str(self.spec["id"])

    @property
    def driver_sha256(self) -> str | None:
        return None if self.spec is None else canonical_sha256(self.spec)

    def _record_prepared(self, details: dict[str, Any]) -> None:
        assert self.spec is not None
        _append_jsonl(
            self.trace_path,
            {
                "event": "boundary_driver_prepared",
                "scenario_id": self.scenario_id,
                "episode_seed": self.episode_seed,
                "driver_id": self.driver_id,
                "driver_sha256": self.driver_sha256,
                "parameters": self.spec["parameters"],
                "recovery_action_provided": False,
                "details": details,
            },
        )

    def reset(self, env: Any) -> None:
        if self.spec is None:
            return
        import torch

        driver_id = str(self.spec["id"])
        parameters = self.spec["parameters"]
        details: dict[str, Any]
        if driver_id == "semantic-hand-close-pulse":
            details = {"mode": "action_seam", "state_modified": False}
        elif driver_id == "stable-object-lift-fixture":
            asset_name = str(parameters["asset_name"])
            asset = env.scene[asset_name]
            state = asset.data.root_state_w[self.env_id : self.env_id + 1].clone()
            state[:, 2] += float(parameters["lift_m"])
            velocity = [
                *[float(value) for value in parameters["linear_velocity_mps"]],
                *[float(value) for value in parameters["angular_velocity_rps"]],
            ]
            state[:, 7:13] = torch.as_tensor(
                velocity, dtype=state.dtype, device=state.device
            ).reshape(1, 6)
            self._target_asset_name = asset_name
            self._target_root_state = state
            self._apply_target_root_state(env)
            details = {
                "mode": "held_scene_state_until_detector_edge",
                "asset_name": asset_name,
                "target_root_state_w": state[0].detach().cpu().tolist(),
            }
        elif driver_id == "door-to-wrist-fixture":
            door_name = str(parameters["door_asset_name"])
            door = env.scene[door_name]
            robot = env.scene["robot"]
            door_names = _body_names(door)
            robot_names = _body_names(robot)
            handle_ids = [
                index for index, name in enumerate(door_names) if "handle" in name.lower()
            ]
            wrist_ids = [
                index
                for index, name in enumerate(robot_names)
                if any(token in name.lower() for token in _WRIST_BODY_TOKENS)
            ]
            if not handle_ids or not wrist_ids:
                raise ValueError("door-to-wrist fixture could not resolve handle/wrist bodies")
            handles = _body_positions(door)[self.env_id, handle_ids, :3]
            wrists = _body_positions(robot)[self.env_id, wrist_ids, :3]
            distances = torch.linalg.vector_norm(
                wrists[:, None, :] - handles[None, :, :], dim=-1
            )
            flat = int(distances.argmin().detach().cpu().item())
            handle_offset = flat % len(handle_ids)
            wrist_offset = flat // len(handle_ids)
            handle = handles[handle_offset]
            wrist = wrists[wrist_offset]
            direction = handle - wrist
            norm = torch.linalg.vector_norm(direction)
            if float(norm.detach().cpu().item()) <= 1e-8:
                direction = torch.tensor(
                    [1.0, 0.0, 0.0], dtype=handle.dtype, device=handle.device
                )
                norm = torch.linalg.vector_norm(direction)
            desired_handle = wrist + direction / norm * float(parameters["target_distance_m"])
            delta = desired_handle - handle
            env_ids = torch.tensor([self.env_id], dtype=torch.long, device=door.device)
            pose = door.data.root_state_w[env_ids, :7].clone()
            pose[:, :3] += delta.reshape(1, 3)
            door.write_root_pose_to_sim(pose, env_ids=env_ids)
            _flush(env)
            details = {
                "mode": "one_shot_scene_state",
                "door_asset_name": door_name,
                "handle_body_name": door_names[handle_ids[handle_offset]],
                "wrist_body_name": robot_names[wrist_ids[wrist_offset]],
                "root_translation_w_m": delta.detach().cpu().tolist(),
                "target_distance_m": float(parameters["target_distance_m"]),
            }
        elif driver_id == "open-door-joint-fixture":
            door_name = str(parameters["door_asset_name"])
            door = env.scene[door_name]
            joint_names = [str(name) for name in door.joint_names]
            matches = [
                index for index, name in enumerate(joint_names)
                if name == str(parameters["leaf_joint_name"])
            ]
            if len(matches) != 1:
                raise ValueError("open-door fixture leaf joint did not resolve exactly once")
            env_ids = torch.tensor([self.env_id], dtype=torch.long, device=door.device)
            positions = door.data.joint_pos[env_ids].clone()
            velocities = torch.zeros_like(door.data.joint_vel[env_ids])
            positions[:, matches[0]] = math.radians(float(parameters["leaf_angle_deg"]))
            self._target_joint_asset_name = door_name
            self._target_joint_positions = positions
            self._target_joint_velocities = velocities
            door.write_joint_state_to_sim(positions, velocities, env_ids=env_ids)
            latch_set = getattr(env.cfg, "_open_door_latch_unlocked_env_ids", None)
            if not isinstance(latch_set, set):
                raise ValueError("open-door fixture cannot access the runtime latch state")
            latch_set.add(self.env_id)
            _flush(env)
            details = {
                "mode": "held_joint_state_until_detector_edge",
                "door_asset_name": door_name,
                "leaf_joint_name": str(parameters["leaf_joint_name"]),
                "leaf_angle_deg": float(parameters["leaf_angle_deg"]),
                "latch_marked_unlocked": True,
            }
        elif driver_id == "object-speed-fixture":
            asset_name = str(parameters["asset_name"])
            asset = env.scene[asset_name]
            env_ids = torch.tensor([self.env_id], dtype=torch.long, device=asset.device)
            velocity = torch.as_tensor(
                [
                    *parameters["linear_velocity_mps"],
                    *parameters["angular_velocity_rps"],
                ],
                dtype=asset.data.root_vel_w.dtype,
                device=asset.device,
            ).reshape(1, 6)
            asset.write_root_velocity_to_sim(velocity, env_ids=env_ids)
            _flush(env)
            details = {
                "mode": "one_shot_scene_state",
                "asset_name": asset_name,
                "target_root_velocity_w": velocity[0].detach().cpu().tolist(),
            }
        elif driver_id in {"object-transport-fixture", "root-displacement-fixture"}:
            asset_name = str(parameters.get("asset_name", "robot"))
            asset = env.scene[asset_name]
            state = asset.data.root_state_w[self.env_id : self.env_id + 1].clone()
            translation = torch.as_tensor(
                parameters["translation_world_m"], dtype=state.dtype, device=state.device
            ).reshape(1, 3)
            state[:, :3] += translation
            state[:, 7:13] = 0.0
            self._target_asset_name = asset_name
            self._target_root_state = state
            self._apply_target_root_state(env)
            details = {
                "mode": "held_scene_state_until_detector_edge",
                "asset_name": asset_name,
                "translation_world_m": translation[0].detach().cpu().tolist(),
                "target_root_state_w": state[0].detach().cpu().tolist(),
            }
        elif driver_id == "seat-approach-fixture":
            rewards = __import__(
                "tasks.g1_tasks.move_sit_sofa_g1_29dof_dex3_wholebody.mdp.rewards",
                fromlist=["rewards"],
            )
            boxes = rewards._get_sofa_seat_boxes_world(env)  # noqa: SLF001
            box = boxes[self.env_id]
            if box is None:
                raise ValueError("seat-approach fixture cannot resolve the live seat AABB")
            body_positions = rewards._get_body_positions_w(env)  # noqa: SLF001
            body_ids = rewards._get_sit_body_indices(env)  # noqa: SLF001
            proxy = body_positions[self.env_id, int(body_ids[0]), :2]
            clearance = float(parameters["target_clearance_m"])
            target_xy = torch.as_tensor(
                [(float(box[0]) + float(box[1])) / 2.0, float(box[3]) + clearance],
                dtype=proxy.dtype,
                device=proxy.device,
            )
            robot = env.scene["robot"]
            state = robot.data.root_state_w[self.env_id : self.env_id + 1].clone()
            delta_xy = target_xy - proxy
            state[:, :2] += delta_xy.reshape(1, 2)
            state[:, 7:13] = 0.0
            self._target_asset_name = "robot"
            self._target_root_state = state
            self._apply_target_root_state(env)
            details = {
                "mode": "held_scene_state_until_detector_edge",
                "asset_name": "robot",
                "target_clearance_m": clearance,
                "root_translation_xy_m": delta_xy.detach().cpu().tolist(),
            }
        elif driver_id == "strike-shell-fixture":
            rewards = __import__(
                "tasks.g1_tasks.move_boxing_bag_g1_29dof_dex3_wholebody.mdp.rewards",
                fromlist=["rewards"],
            )
            body_positions = rewards._get_body_positions_w(env)  # noqa: SLF001
            body_ids = rewards._get_punch_body_indices(env)  # noqa: SLF001
            punches = body_positions[self.env_id, list(body_ids), :3]
            target = rewards._get_boxing_target_positions_world(env)[self.env_id]  # noqa: SLF001
            threshold = rewards._get_boxing_target_hit_distance_thresholds(env)[  # noqa: SLF001
                self.env_id
            ]
            distances = torch.linalg.vector_norm(punches - target.reshape(1, 3), dim=-1)
            punch = punches[int(distances.argmin().detach().cpu().item())]
            direction = punch - target
            norm = torch.linalg.vector_norm(direction)
            if float(norm.detach().cpu().item()) <= 1e-8:
                direction = torch.tensor(
                    [1.0, 0.0, 0.0], dtype=punch.dtype, device=punch.device
                )
                norm = torch.linalg.vector_norm(direction)
            unit = direction / norm
            robot = env.scene["robot"]
            base_state = robot.data.root_state_w[self.env_id : self.env_id + 1].clone()
            for clearance in parameters["clearance_schedule_m"]:
                desired = target + unit * (float(threshold) + float(clearance))
                state = base_state.clone()
                state[:, :3] += (desired - punch).reshape(1, 3)
                state[:, 7:13] = 0.0
                self._target_root_schedule.append(state)
            self._target_asset_name = "robot"
            self._target_root_state = self._target_root_schedule[0]
            self._apply_target_root_state(env)
            details = {
                "mode": "scheduled_scene_state_until_detector_edge",
                "asset_name": "robot",
                "clearance_schedule_m": [
                    float(value) for value in parameters["clearance_schedule_m"]
                ],
            }
        else:
            raise ValueError(f"unsupported boundary driver: {driver_id}")
        self._record_prepared(details)

    def _apply_target_root_state(self, env: Any) -> None:
        if self._target_root_state is None or self._target_asset_name is None:
            return
        import torch

        asset = env.scene[self._target_asset_name]
        env_ids = torch.tensor([self.env_id], dtype=torch.long, device=asset.device)
        _write_root_state(asset, self._target_root_state, env_ids)
        _flush(env)

    def _apply_target_joint_state(self, env: Any) -> None:
        if (
            self._target_joint_asset_name is None
            or self._target_joint_positions is None
            or self._target_joint_velocities is None
        ):
            return
        import torch

        asset = env.scene[self._target_joint_asset_name]
        env_ids = torch.tensor([self.env_id], dtype=torch.long, device=asset.device)
        asset.write_joint_state_to_sim(
            self._target_joint_positions,
            self._target_joint_velocities,
            env_ids=env_ids,
        )
        _flush(env)

    def before_observe(self, env: Any) -> None:
        if self.active:
            if self._target_root_schedule:
                index = min(
                    self._target_root_schedule_index,
                    len(self._target_root_schedule) - 1,
                )
                self._target_root_state = self._target_root_schedule[index]
                self._target_root_schedule_index += 1
            self._apply_target_root_state(env)
            self._apply_target_joint_state(env)

    def transform_semantic_action(self, action: Any) -> Any:
        if not self.active or self.driver_id != "semantic-hand-close-pulse":
            return action
        import numpy as np

        assert self.spec is not None
        parameters = self.spec["parameters"]
        if self._action_samples >= int(parameters["max_action_samples"]):
            return action
        before = np.asarray(action, dtype=np.float32)
        if before.shape[-1] != 40:
            raise ValueError("semantic hand-close boundary driver requires action40")
        after = before.copy()
        for index in parameters["hand_indices"]:
            after[..., int(index)] = float(parameters["value"])
        self._action_samples += 1
        _append_jsonl(
            self.trace_path,
            {
                "event": "boundary_driver_action_pulse",
                "scenario_id": self.scenario_id,
                "episode_seed": self.episode_seed,
                "driver_id": self.driver_id,
                "driver_sha256": self.driver_sha256,
                "sample_index": self._action_samples,
                "input_sha256": canonical_sha256(before.tolist()),
                "output_sha256": canonical_sha256(after.tolist()),
                "recovery_action_provided": False,
            },
        )
        return after

    def stop_at_detector_edge(self) -> None:
        self.active = False


__all__ = ["DeterministicBoundaryDriver"]
