"""Static contracts for the nine HumanoidArena recovery perturbations.

This module validates that every draft scenario names an observable simulator
boundary and a fully specified physical/action perturbation.  Passing this
check is intentionally weaker than runtime admission: no injector is credited
until its live trace, snapshots, and independent oracle trials exist.
"""

from __future__ import annotations

import math
from typing import Any

from .plan import canonical_sha256


SCENARIO_BINDINGS = {
    "box-missed-grasp-retry": (
        "pick_and_place_box",
        "online_failure",
        "first-hand-close",
        "release-grasp-contact",
        "semantic-action40",
    ),
    "box-drop-and-body-push": (
        "pick_and_place_box",
        "failure_start",
        "stable-object-lift",
        "drop-object-and-root-velocity",
        "scene+semantic-action40",
    ),
    "door-handle-pose-shift": (
        "open_door",
        "online_failure",
        "wrist-handle-approach",
        "shift-door-handle",
        "scene",
    ),
    "doorway-obstruction": (
        "open_door",
        "failure_start",
        "door-open-before-traversal",
        "place-doorway-obstacle",
        "scene",
    ),
    "tracker-underexecution": (
        "double_desk",
        "online_failure",
        "object-in-transport",
        "attenuate-sonic-latent",
        "post-encoder-latent64",
    ),
    "support-state-push": (
        "football",
        "online_failure",
        "object-first-motion",
        "root-velocity-impulse",
        "scene",
    ),
    "sofa-approach-slip": (
        "sit_sofa",
        "failure_start",
        "seat-approach",
        "support-foot-slip",
        "scene",
    ),
    "contact-recoil": (
        "boxing",
        "online_failure",
        "strike-approach-shell",
        "upper-body-contact-impulse",
        "scene",
    ),
    "new-blocking-obstacle": (
        "visual_navigation",
        "online_failure",
        "root-displacement",
        "place-path-obstacle",
        "scene",
    ),
}


def _exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label}: keys are {actual}, expected {sorted(keys)}")
    return value


def _number(value: Any, label: str, *, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: must be numeric")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{label}: must be finite and in [{low}, {high}]")
    return result


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label}: must be a non-empty canonical string")
    return value


def _choice(value: Any, allowed: set[str], label: str) -> str:
    result = _name(value, label)
    if result not in allowed:
        raise ValueError(f"{label}: {result!r} is not one of {sorted(allowed)}")
    return result


def _vector(value: Any, width: int, label: str, *, low: float, high: float) -> list[float]:
    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"{label}: must contain exactly {width} values")
    return [
        _number(item, f"{label}[{index}]", low=low, high=high)
        for index, item in enumerate(value)
    ]


def _validate_detector(detector_id: str, parameters: Any, label: str) -> None:
    if detector_id == "first-hand-close":
        p = _exact_keys(parameters, {"close_threshold", "hand_indices"}, label)
        _number(p["close_threshold"], f"{label}.close_threshold", low=0.0, high=1.0)
        if p["hand_indices"] != [38, 39]:
            raise ValueError(f"{label}.hand_indices: semantic-action40 hands must be [38, 39]")
    elif detector_id == "stable-object-lift":
        p = _exact_keys(
            parameters,
            {"asset_name", "minimum_lift_m", "maximum_speed_mps", "stable_duration_s"},
            label,
        )
        _name(p["asset_name"], f"{label}.asset_name")
        _number(p["minimum_lift_m"], f"{label}.minimum_lift_m", low=0.01, high=1.0)
        _number(p["maximum_speed_mps"], f"{label}.maximum_speed_mps", low=0.0, high=2.0)
        _number(p["stable_duration_s"], f"{label}.stable_duration_s", low=0.1, high=5.0)
    elif detector_id == "wrist-handle-approach":
        p = _exact_keys(parameters, {"door_asset_name", "maximum_distance_m"}, label)
        _name(p["door_asset_name"], f"{label}.door_asset_name")
        _number(p["maximum_distance_m"], f"{label}.maximum_distance_m", low=0.05, high=1.0)
    elif detector_id == "door-open-before-traversal":
        p = _exact_keys(parameters, {"minimum_leaf_angle_deg"}, label)
        _number(p["minimum_leaf_angle_deg"], f"{label}.minimum_leaf_angle_deg", low=1.0, high=120.0)
    elif detector_id == "object-in-transport":
        p = _exact_keys(
            parameters, {"asset_name", "minimum_displacement_m", "minimum_lift_m"}, label
        )
        _name(p["asset_name"], f"{label}.asset_name")
        _number(
            p["minimum_displacement_m"], f"{label}.minimum_displacement_m", low=0.01, high=5.0
        )
        _number(p["minimum_lift_m"], f"{label}.minimum_lift_m", low=0.01, high=1.0)
    elif detector_id == "object-first-motion":
        p = _exact_keys(parameters, {"asset_name", "minimum_speed_mps"}, label)
        _name(p["asset_name"], f"{label}.asset_name")
        _number(p["minimum_speed_mps"], f"{label}.minimum_speed_mps", low=0.01, high=10.0)
    elif detector_id == "seat-approach":
        p = _exact_keys(parameters, {"maximum_xy_distance_m"}, label)
        _number(
            p["maximum_xy_distance_m"], f"{label}.maximum_xy_distance_m", low=0.05, high=3.0
        )
    elif detector_id == "strike-approach-shell":
        p = _exact_keys(
            parameters, {"inner_margin_m", "outer_margin_m", "decreasing_samples"}, label
        )
        inner = _number(p["inner_margin_m"], f"{label}.inner_margin_m", low=0.0, high=1.0)
        outer = _number(p["outer_margin_m"], f"{label}.outer_margin_m", low=0.0, high=1.0)
        if not inner < outer:
            raise ValueError(f"{label}: inner margin must be smaller than outer margin")
        if type(p["decreasing_samples"]) is not int or not 2 <= p["decreasing_samples"] <= 20:
            raise ValueError(f"{label}.decreasing_samples: must be an integer in [2, 20]")
    elif detector_id == "root-displacement":
        p = _exact_keys(parameters, {"minimum_xy_displacement_m"}, label)
        _number(
            p["minimum_xy_displacement_m"],
            f"{label}.minimum_xy_displacement_m",
            low=0.1,
            high=20.0,
        )
    else:
        raise ValueError(f"{label}: unsupported detector {detector_id!r}")


def _validate_injector(injector_id: str, parameters: Any, label: str) -> None:
    if injector_id == "release-grasp-contact":
        p = _exact_keys(parameters, {"duration_s"}, label)
        _number(p["duration_s"], f"{label}.duration_s", low=0.02, high=2.0)
    elif injector_id == "drop-object-and-root-velocity":
        p = _exact_keys(
            parameters,
            {
                "object_asset_name",
                "release_duration_s",
                "lateral_mps",
                "lateral_direction_robot",
            },
            label,
        )
        _name(p["object_asset_name"], f"{label}.object_asset_name")
        _number(p["release_duration_s"], f"{label}.release_duration_s", low=0.02, high=2.0)
        _number(p["lateral_mps"], f"{label}.lateral_mps", low=0.05, high=3.0)
        _choice(p["lateral_direction_robot"], {"left", "right"}, f"{label}.lateral_direction_robot")
    elif injector_id == "shift-door-handle":
        p = _exact_keys(
            parameters, {"door_asset_name", "translation_m", "translation_axis_door_local"}, label
        )
        _name(p["door_asset_name"], f"{label}.door_asset_name")
        _number(p["translation_m"], f"{label}.translation_m", low=0.005, high=0.2)
        _choice(
            p["translation_axis_door_local"], {"x", "y"}, f"{label}.translation_axis_door_local"
        )
    elif injector_id == "place-doorway-obstacle":
        p = _exact_keys(parameters, {"mass_kg", "size_m", "door_local_center_m"}, label)
        _number(p["mass_kg"], f"{label}.mass_kg", low=0.05, high=20.0)
        size = _vector(p["size_m"], 3, f"{label}.size_m", low=0.05, high=2.0)
        if any(value <= 0.0 for value in size):
            raise ValueError(f"{label}.size_m: dimensions must be positive")
        _vector(p["door_local_center_m"], 3, f"{label}.door_local_center_m", low=-5.0, high=5.0)
    elif injector_id == "attenuate-sonic-latent":
        p = _exact_keys(parameters, {"scale", "duration_s"}, label)
        _number(p["scale"], f"{label}.scale", low=0.0, high=0.999999)
        _number(p["duration_s"], f"{label}.duration_s", low=0.02, high=5.0)
    elif injector_id == "root-velocity-impulse":
        p = _exact_keys(parameters, {"lateral_mps", "lateral_direction_robot"}, label)
        _number(p["lateral_mps"], f"{label}.lateral_mps", low=0.05, high=3.0)
        _choice(p["lateral_direction_robot"], {"left", "right"}, f"{label}.lateral_direction_robot")
    elif injector_id == "support-foot-slip":
        p = _exact_keys(parameters, {"displacement_m", "lateral_direction_robot"}, label)
        _number(p["displacement_m"], f"{label}.displacement_m", low=0.01, high=0.5)
        _choice(p["lateral_direction_robot"], {"left", "right"}, f"{label}.lateral_direction_robot")
    elif injector_id == "upper-body-contact-impulse":
        p = _exact_keys(parameters, {"impulse_ns", "body_name", "direction"}, label)
        _number(p["impulse_ns"], f"{label}.impulse_ns", low=0.1, high=200.0)
        _name(p["body_name"], f"{label}.body_name")
        _choice(p["direction"], {"away-from-target"}, f"{label}.direction")
    elif injector_id == "place-path-obstacle":
        p = _exact_keys(
            parameters, {"clearance_m", "asset_name", "distance_ahead_m"}, label
        )
        _number(p["clearance_m"], f"{label}.clearance_m", low=0.0, high=2.0)
        _name(p["asset_name"], f"{label}.asset_name")
        _number(p["distance_ahead_m"], f"{label}.distance_ahead_m", low=0.2, high=5.0)
    else:
        raise ValueError(f"{label}: unsupported injector {injector_id!r}")


def validate_recovery_injector_contract(suite: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate complete, typed event/injector bindings for the locked v0 suite."""

    if suite.get("suite_id") != "hrvla_recovery_v0":
        raise ValueError("injector contract only applies to hrvla_recovery_v0")
    scenarios: dict[str, tuple[str, dict[str, Any]]] = {}
    for task in suite.get("tasks", []):
        task_id = str(task.get("id", ""))
        for scenario in task.get("scenarios", []):
            scenario_id = str(scenario.get("id", ""))
            if scenario_id in scenarios:
                raise ValueError(f"duplicate recovery scenario {scenario_id!r}")
            scenarios[scenario_id] = (task_id, scenario)
    if set(scenarios) != set(SCENARIO_BINDINGS):
        raise ValueError(
            "recovery scenario set differs from the locked nine: "
            f"actual={sorted(scenarios)}"
        )

    open_door = next(task for task in suite["tasks"] if task["id"] == "open_door")
    if open_door.get("runtime_environment") != {"OPEN_DOOR_STRICT_REQUIRE_GEOMETRY": "1"}:
        raise ValueError("open_door recovery must require geometry to make obstruction measurable")

    rows = []
    for scenario_id, binding in SCENARIO_BINDINGS.items():
        expected_task, expected_protocol, detector_id, injector_id, seam = binding
        task_id, scenario = scenarios[scenario_id]
        if task_id != expected_task or scenario.get("protocol") != expected_protocol:
            raise ValueError(f"{scenario_id}: task or protocol differs from the locked binding")
        detector = _exact_keys(scenario.get("event_detector"), {"id", "parameters"}, f"{scenario_id}.event_detector")
        injector = _exact_keys(scenario.get("injector"), {"id", "parameters"}, f"{scenario_id}.injector")
        if detector["id"] != detector_id or injector["id"] != injector_id:
            raise ValueError(f"{scenario_id}: detector or injector differs from the locked binding")
        if not isinstance(scenario.get("event_boundary"), str) or not scenario["event_boundary"].strip():
            raise ValueError(f"{scenario_id}: event_boundary is required")
        _validate_detector(detector_id, detector["parameters"], f"{scenario_id}.event_detector")
        _validate_injector(injector_id, injector["parameters"], f"{scenario_id}.injector")
        rows.append(
            {
                "task_id": task_id,
                "scenario_id": scenario_id,
                "protocol": expected_protocol,
                "event_id": scenario["event_id"],
                "detector_id": detector_id,
                "detector_parameters_sha256": canonical_sha256(detector["parameters"]),
                "injector_id": injector_id,
                "injector_parameters_sha256": canonical_sha256(injector["parameters"]),
                "interface_seam": seam,
            }
        )
    return rows


__all__ = ["SCENARIO_BINDINGS", "validate_recovery_injector_contract"]
