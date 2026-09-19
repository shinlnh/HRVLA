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
        "shift-door-assembly-handle-frame",
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
        "root-lateral-velocity-delta",
        "scene",
    ),
    "sofa-approach-slip": (
        "sit_sofa",
        "failure_start",
        "seat-approach",
        "support-foot-lateral-impulse",
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


BOUNDARY_DRIVER_BINDINGS = {
    "box-missed-grasp-retry": "semantic-hand-close-pulse",
    "box-drop-and-body-push": "stable-object-lift-fixture",
    "door-handle-pose-shift": "door-to-wrist-fixture",
    "doorway-obstruction": "open-door-joint-fixture",
    "support-state-push": "object-speed-fixture",
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
        p = _exact_keys(
            parameters,
            {"maximum_xy_distance_m", "minimum_root_height_m", "minimum_up_axis_z"},
            label,
        )
        _number(
            p["maximum_xy_distance_m"], f"{label}.maximum_xy_distance_m", low=0.05, high=3.0
        )
        _number(
            p["minimum_root_height_m"], f"{label}.minimum_root_height_m", low=0.1, high=2.0
        )
        _number(p["minimum_up_axis_z"], f"{label}.minimum_up_axis_z", low=0.0, high=1.0)
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
    elif injector_id == "shift-door-assembly-handle-frame":
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
    elif injector_id == "root-lateral-velocity-delta":
        p = _exact_keys(parameters, {"lateral_mps", "lateral_direction_robot"}, label)
        _number(p["lateral_mps"], f"{label}.lateral_mps", low=0.05, high=3.0)
        _choice(p["lateral_direction_robot"], {"left", "right"}, f"{label}.lateral_direction_robot")
    elif injector_id == "support-foot-lateral-impulse":
        p = _exact_keys(
            parameters,
            {"impulse_ns", "lateral_direction_robot", "body_selection"},
            label,
        )
        _number(p["impulse_ns"], f"{label}.impulse_ns", low=0.1, high=100.0)
        _choice(p["lateral_direction_robot"], {"left", "right"}, f"{label}.lateral_direction_robot")
        _choice(
            p["body_selection"],
            {"max-contact-ankle-roll"},
            f"{label}.body_selection",
        )
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


def _validate_boundary_driver(driver_id: str, parameters: Any, label: str) -> None:
    if driver_id == "semantic-hand-close-pulse":
        p = _exact_keys(parameters, {"hand_indices", "value", "max_action_samples"}, label)
        if p["hand_indices"] != [38, 39]:
            raise ValueError(f"{label}.hand_indices: semantic-action40 hands must be [38, 39]")
        _number(p["value"], f"{label}.value", low=0.5, high=1.0)
        if p["max_action_samples"] != 1:
            raise ValueError(f"{label}.max_action_samples: must be exactly one")
    elif driver_id == "stable-object-lift-fixture":
        p = _exact_keys(
            parameters,
            {"asset_name", "lift_m", "linear_velocity_mps", "angular_velocity_rps"},
            label,
        )
        _name(p["asset_name"], f"{label}.asset_name")
        _number(p["lift_m"], f"{label}.lift_m", low=0.05, high=0.25)
        _vector(p["linear_velocity_mps"], 3, f"{label}.linear_velocity_mps", low=-1.0, high=1.0)
        _vector(p["angular_velocity_rps"], 3, f"{label}.angular_velocity_rps", low=-1.0, high=1.0)
    elif driver_id == "door-to-wrist-fixture":
        p = _exact_keys(parameters, {"door_asset_name", "target_distance_m"}, label)
        _name(p["door_asset_name"], f"{label}.door_asset_name")
        _number(p["target_distance_m"], f"{label}.target_distance_m", low=0.1, high=0.3)
    elif driver_id == "open-door-joint-fixture":
        p = _exact_keys(
            parameters,
            {"door_asset_name", "leaf_joint_name", "leaf_angle_deg", "mark_latch_unlocked"},
            label,
        )
        _name(p["door_asset_name"], f"{label}.door_asset_name")
        _name(p["leaf_joint_name"], f"{label}.leaf_joint_name")
        _number(p["leaf_angle_deg"], f"{label}.leaf_angle_deg", low=60.0, high=90.0)
        if p["mark_latch_unlocked"] is not True:
            raise ValueError(f"{label}.mark_latch_unlocked: must be true")
    elif driver_id == "object-speed-fixture":
        p = _exact_keys(
            parameters,
            {"asset_name", "linear_velocity_mps", "angular_velocity_rps"},
            label,
        )
        _name(p["asset_name"], f"{label}.asset_name")
        linear = _vector(
            p["linear_velocity_mps"], 3, f"{label}.linear_velocity_mps", low=-2.0, high=2.0
        )
        if math.sqrt(sum(value * value for value in linear)) < 0.25:
            raise ValueError(f"{label}.linear_velocity_mps: norm must be at least 0.25")
        _vector(p["angular_velocity_rps"], 3, f"{label}.angular_velocity_rps", low=-2.0, high=2.0)
    else:
        raise ValueError(f"{label}: unsupported boundary driver {driver_id!r}")


def validate_recovery_injector_contract(suite: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate complete, typed event/injector bindings for the locked v0 suite."""

    if suite.get("suite_id") != "hrvla_recovery_v0":
        raise ValueError("injector contract only applies to hrvla_recovery_v0")
    capture = _exact_keys(
        suite.get("admission_capture"),
        {
            "snapshot_seed",
            "failure_snapshot_policy",
            "boundary_reachability_protocol",
            "boundary_driver_selection_rule",
            "boundary_driver_claim_boundary",
        },
        "admission_capture",
    )
    if type(capture["snapshot_seed"]) is not int or capture["snapshot_seed"] < 0:
        raise ValueError("admission_capture.snapshot_seed must be a non-negative integer")
    _name(capture["failure_snapshot_policy"], "admission_capture.failure_snapshot_policy")
    if capture["boundary_reachability_protocol"] != "pre_registered_deterministic_driver_v1":
        raise ValueError("admission_capture boundary reachability protocol differs")
    _name(capture["boundary_driver_selection_rule"], "admission_capture.boundary_driver_selection_rule")
    _name(capture["boundary_driver_claim_boundary"], "admission_capture.boundary_driver_claim_boundary")
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
        if expected_protocol == "failure_start":
            _number(
                scenario.get("failure_snapshot_settle_s"),
                f"{scenario_id}.failure_snapshot_settle_s",
                low=0.02,
                high=5.0,
            )
        elif "failure_snapshot_settle_s" in scenario:
            raise ValueError(
                f"{scenario_id}: online failure must not define a failure snapshot settle interval"
            )
        detector = _exact_keys(scenario.get("event_detector"), {"id", "parameters"}, f"{scenario_id}.event_detector")
        injector = _exact_keys(scenario.get("injector"), {"id", "parameters"}, f"{scenario_id}.injector")
        if detector["id"] != detector_id or injector["id"] != injector_id:
            raise ValueError(f"{scenario_id}: detector or injector differs from the locked binding")
        if not isinstance(scenario.get("event_boundary"), str) or not scenario["event_boundary"].strip():
            raise ValueError(f"{scenario_id}: event_boundary is required")
        _validate_detector(detector_id, detector["parameters"], f"{scenario_id}.event_detector")
        _validate_injector(injector_id, injector["parameters"], f"{scenario_id}.injector")
        driver = scenario.get("boundary_driver")
        expected_driver = BOUNDARY_DRIVER_BINDINGS.get(scenario_id)
        if expected_driver is None:
            if driver is not None:
                raise ValueError(f"{scenario_id}: natural boundary scenario must not define a driver")
            driver_sha256 = None
        else:
            driver = _exact_keys(driver, {"id", "parameters"}, f"{scenario_id}.boundary_driver")
            if driver["id"] != expected_driver:
                raise ValueError(f"{scenario_id}: boundary driver differs from the locked binding")
            _validate_boundary_driver(
                expected_driver, driver["parameters"], f"{scenario_id}.boundary_driver"
            )
            driver_sha256 = canonical_sha256(driver)
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
                "boundary_driver_id": expected_driver,
                "boundary_driver_sha256": driver_sha256,
            }
        )
    return rows


__all__ = [
    "BOUNDARY_DRIVER_BINDINGS",
    "SCENARIO_BINDINGS",
    "validate_recovery_injector_contract",
]
