"""Stateful semantic-boundary detectors for single-environment recovery runs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Mapping


def _values(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def _root_pose(asset: Any, env_id: int) -> list[float]:
    pose = getattr(asset.data, "root_link_pose_w", None)
    if pose is not None:
        return _values(pose[env_id])
    state = getattr(asset.data, "root_state_w", None)
    if state is None:
        raise ValueError("asset does not expose root_link_pose_w or root_state_w")
    return _values(state[env_id][:7])


def _root_velocity(asset: Any, env_id: int) -> list[float]:
    velocity = getattr(asset.data, "root_vel_w", None)
    if velocity is not None:
        return _values(velocity[env_id])
    state = getattr(asset.data, "root_state_w", None)
    if state is None:
        raise ValueError("asset does not expose root_vel_w or root_state_w")
    return _values(state[env_id][7:13])


def _finite_number(signals: Mapping[str, Any], name: str) -> float:
    value = signals.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"runtime signal {name!r} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"runtime signal {name!r} must be finite")
    return result


@dataclass(frozen=True)
class EventObservation:
    triggered: bool
    detector_id: str
    sample_index: int
    diagnostics: dict[str, Any]


class SemanticEventDetector:
    """Evaluate one locked event boundary and latch its first rising edge."""

    def __init__(
        self,
        detector: Mapping[str, Any],
        *,
        control_dt_s: float,
        env_id: int = 0,
    ) -> None:
        if control_dt_s <= 0.0 or not math.isfinite(control_dt_s):
            raise ValueError("control_dt_s must be finite and positive")
        self.detector_id = str(detector["id"])
        self.parameters = dict(detector["parameters"])
        self.control_dt_s = float(control_dt_s)
        self.env_id = int(env_id)
        self._initialized = False
        self._triggered = False
        self._sample_index = 0
        self._initial_asset_position: list[float] | None = None
        self._initial_robot_position: list[float] | None = None
        self._hand_closed = False
        self._stable_samples = 0
        self._strike_clearance = deque(maxlen=20)

    def reset(self, env: Any) -> None:
        self._initialized = True
        self._triggered = False
        self._sample_index = 0
        self._initial_asset_position = None
        self._initial_robot_position = _root_pose(env.scene["robot"], self.env_id)[:3]
        asset_name = self.parameters.get("asset_name")
        if asset_name:
            self._initial_asset_position = _root_pose(
                env.scene[str(asset_name)], self.env_id
            )[:3]
        self._hand_closed = False
        self._stable_samples = 0
        self._strike_clearance.clear()

    def _first_hand_close(self, action: Any) -> tuple[bool, dict[str, Any]]:
        if action is None:
            raise ValueError("first-hand-close requires semantic_action40")
        values = _values(action)
        indices = self.parameters["hand_indices"]
        if len(values) != 40 or indices != [38, 39]:
            raise ValueError("first-hand-close requires semantic action40 hand indices [38, 39]")
        threshold = float(self.parameters["close_threshold"])
        closed = any(values[index] >= threshold for index in indices)
        rising = closed and not self._hand_closed
        self._hand_closed = closed
        return rising, {"hand_values": [values[index] for index in indices], "closed": closed}

    def _stable_object_lift(self, env: Any) -> tuple[bool, dict[str, Any]]:
        name = str(self.parameters["asset_name"])
        position = _root_pose(env.scene[name], self.env_id)[:3]
        velocity = _root_velocity(env.scene[name], self.env_id)[:3]
        assert self._initial_asset_position is not None
        lift = position[2] - self._initial_asset_position[2]
        speed = math.sqrt(sum(value * value for value in velocity))
        stable = (
            lift >= float(self.parameters["minimum_lift_m"])
            and speed <= float(self.parameters["maximum_speed_mps"])
        )
        self._stable_samples = self._stable_samples + 1 if stable else 0
        required = math.ceil(float(self.parameters["stable_duration_s"]) / self.control_dt_s)
        return self._stable_samples >= required, {
            "lift_m": lift,
            "speed_mps": speed,
            "stable_samples": self._stable_samples,
            "required_stable_samples": required,
        }

    def _object_in_transport(self, env: Any) -> tuple[bool, dict[str, Any]]:
        name = str(self.parameters["asset_name"])
        position = _root_pose(env.scene[name], self.env_id)[:3]
        assert self._initial_asset_position is not None
        delta = [value - base for value, base in zip(position, self._initial_asset_position)]
        displacement = math.sqrt(sum(value * value for value in delta))
        lift = delta[2]
        triggered = (
            displacement >= float(self.parameters["minimum_displacement_m"])
            and lift >= float(self.parameters["minimum_lift_m"])
        )
        return triggered, {"displacement_m": displacement, "lift_m": lift}

    def _object_first_motion(self, env: Any) -> tuple[bool, dict[str, Any]]:
        name = str(self.parameters["asset_name"])
        velocity = _root_velocity(env.scene[name], self.env_id)[:3]
        speed = math.sqrt(sum(value * value for value in velocity))
        return speed >= float(self.parameters["minimum_speed_mps"]), {"speed_mps": speed}

    def _root_displacement(self, env: Any) -> tuple[bool, dict[str, Any]]:
        position = _root_pose(env.scene["robot"], self.env_id)[:3]
        assert self._initial_robot_position is not None
        dx = position[0] - self._initial_robot_position[0]
        dy = position[1] - self._initial_robot_position[1]
        distance = math.hypot(dx, dy)
        return distance >= float(self.parameters["minimum_xy_displacement_m"]), {
            "xy_displacement_m": distance
        }

    def _external_signal(self, signals: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
        if self.detector_id == "wrist-handle-approach":
            distance = _finite_number(signals, "wrist_handle_min_distance_m")
            return distance <= float(self.parameters["maximum_distance_m"]), {
                "wrist_handle_min_distance_m": distance
            }
        if self.detector_id == "door-open-before-traversal":
            angle = abs(_finite_number(signals, "door_leaf_angle_deg"))
            latch = signals.get("door_latch_unlocked")
            if not isinstance(latch, bool):
                raise ValueError("runtime signal 'door_latch_unlocked' must be boolean")
            return latch and angle >= float(self.parameters["minimum_leaf_angle_deg"]), {
                "door_leaf_angle_deg": angle,
                "door_latch_unlocked": latch,
            }
        if self.detector_id == "seat-approach":
            distance = _finite_number(signals, "seat_xy_distance_m")
            upright = signals.get("robot_upright")
            if not isinstance(upright, bool):
                raise ValueError("runtime signal 'robot_upright' must be boolean")
            return upright and distance <= float(self.parameters["maximum_xy_distance_m"]), {
                "seat_xy_distance_m": distance,
                "robot_upright": upright,
            }
        if self.detector_id == "strike-approach-shell":
            clearance = _finite_number(signals, "punch_hit_clearance_m")
            self._strike_clearance.append(clearance)
            count = int(self.parameters["decreasing_samples"])
            recent = list(self._strike_clearance)[-count:]
            decreasing = len(recent) == count and all(
                later < earlier for earlier, later in zip(recent, recent[1:])
            )
            inside = (
                float(self.parameters["inner_margin_m"])
                <= clearance
                <= float(self.parameters["outer_margin_m"])
            )
            return decreasing and inside, {
                "punch_hit_clearance_m": clearance,
                "recent_clearance_m": recent,
                "strictly_decreasing": decreasing,
            }
        raise ValueError(f"detector {self.detector_id!r} has no external-signal implementation")

    def observe(
        self,
        env: Any,
        *,
        task_success: bool,
        semantic_action: Any = None,
        signals: Mapping[str, Any] | None = None,
    ) -> EventObservation:
        if not self._initialized:
            raise RuntimeError("detector.reset(env) is required before observe")
        self._sample_index += 1
        if self._triggered or task_success:
            return EventObservation(
                False,
                self.detector_id,
                self._sample_index,
                {"latched": self._triggered, "task_success": bool(task_success)},
            )

        if self.detector_id == "first-hand-close":
            triggered, diagnostics = self._first_hand_close(semantic_action)
        elif self.detector_id == "stable-object-lift":
            triggered, diagnostics = self._stable_object_lift(env)
        elif self.detector_id == "object-in-transport":
            triggered, diagnostics = self._object_in_transport(env)
        elif self.detector_id == "object-first-motion":
            triggered, diagnostics = self._object_first_motion(env)
        elif self.detector_id == "root-displacement":
            triggered, diagnostics = self._root_displacement(env)
        else:
            triggered, diagnostics = self._external_signal(signals or {})
        self._triggered = bool(triggered)
        return EventObservation(
            bool(triggered), self.detector_id, self._sample_index, diagnostics
        )


__all__ = ["EventObservation", "SemanticEventDetector"]
