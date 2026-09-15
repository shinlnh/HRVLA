"""Runtime coordinator for audited HumanoidArena recovery perturbations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .humanoidarena_recovery_signals import signals_for_detector
from .isaac_events import (
    apply_action_window,
    apply_asset_local_translation_once,
    apply_body_impulse_once,
    apply_root_local_lateral_velocity_once,
    clear_expired_body_impulses,
    local_lateral_vector_world,
    place_asset_relative_once,
    reset_one_shot_injectors,
)
from .isaac_snapshot import (
    capture_scene_snapshot,
    load_snapshot,
    write_snapshot_atomic,
)
from .plan import canonical_sha256
from .recovery_event_detector import EventObservation, SemanticEventDetector
from .recovery_injector_contract import validate_recovery_injector_contract


TASK_MUTABLE_ASSETS = {
    "pick_and_place_box": ("robot", "box"),
    "open_door": ("robot", "door", "hrvla_doorway_obstacle"),
    "double_desk": ("robot", "object_l"),
    "football": ("robot", "object"),
    "sit_sofa": ("robot", "carton_obstacle_01", "carton_obstacle_02"),
    "boxing": ("robot",),
    "visual_navigation": (
        "robot",
        "obstacle_01_a",
        "obstacle_01_b",
        "obstacle_01_c",
        "obstacle_02_a",
        "obstacle_02_b",
        "obstacle_02_c",
    ),
}


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _write_snapshot_once(path: Path, snapshot: dict[str, Any]) -> str:
    expected = snapshot["snapshot_sha256"]
    if path.exists():
        existing = load_snapshot(path)
        if existing["snapshot_sha256"] != expected:
            raise RuntimeError(
                f"refusing to replace immutable snapshot {path}: "
                f"{existing['snapshot_sha256']} != {expected}"
            )
        return expected
    write_snapshot_atomic(path, snapshot)
    return expected


def configure_recovery_scene(env_cfg: Any, suite: dict[str, Any], task_id: str) -> None:
    """Add deterministic recovery-only assets before ``gym.make`` constructs the scene."""

    if task_id != "open_door":
        return
    task = next((item for item in suite["tasks"] if item["id"] == task_id), None)
    if task is None:
        raise ValueError(f"task {task_id!r} is absent from the recovery suite")
    obstruction = next(
        item
        for item in task["scenarios"]
        if item["injector"]["id"] == "place-doorway-obstacle"
    )
    parameters = obstruction["injector"]["parameters"]

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

    if getattr(env_cfg.scene, "hrvla_doorway_obstacle", None) is not None:
        raise ValueError("recovery doorway obstacle is already configured")
    size = tuple(float(value) for value in parameters["size_m"])
    env_cfg.scene.hrvla_obstacle_parking = AssetBaseCfg(
        prim_path="/World/envs/env_.*/HRVLAObstacleParking",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-50.0, -50.0, -0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 2.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)),
        ),
    )
    env_cfg.scene.hrvla_doorway_obstacle = RigidObjectCfg(
        prim_path="/World/envs/env_.*/HRVLADoorwayObstacle",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-50.0, -50.0, size[2] * 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(parameters["mass_kg"])),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.12, 0.12)),
        ),
    )


class HumanoidArenaRecoveryRuntime:
    """Join a locked detector, injector, trace, and optional snapshot capture."""

    def __init__(
        self,
        suite: dict[str, Any],
        scenario_id: str,
        *,
        control_dt_s: float,
        simulator_revision: str,
        implementation_revision: str,
        output_dir: Path,
        capture_initial_snapshot: bool = False,
        capture_failure_snapshot: bool = False,
        start_snapshot_sha256: str | None = None,
        restore_audit_sha256: str | None = None,
        env_id: int = 0,
    ) -> None:
        contract_rows = validate_recovery_injector_contract(suite)
        contract = next(
            (row for row in contract_rows if row["scenario_id"] == scenario_id), None
        )
        if contract is None:
            raise ValueError(f"unknown recovery scenario: {scenario_id}")
        task = next(task for task in suite["tasks"] if task["id"] == contract["task_id"])
        scenario = next(item for item in task["scenarios"] if item["id"] == scenario_id)
        if capture_failure_snapshot and scenario["protocol"] != "failure_start":
            raise ValueError("failure snapshots may only be captured for failure_start scenarios")
        if not math.isfinite(control_dt_s) or control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be finite and positive")
        if not simulator_revision:
            raise ValueError("simulator_revision is required")
        if (
            len(implementation_revision) != 40
            or set(implementation_revision) - set("0123456789abcdef")
        ):
            raise ValueError("implementation_revision must be a lowercase Git SHA")
        if (start_snapshot_sha256 is None) != (restore_audit_sha256 is None):
            raise ValueError("start snapshot and restore audit hashes must be supplied together")
        for label, value in (
            ("start_snapshot_sha256", start_snapshot_sha256),
            ("restore_audit_sha256", restore_audit_sha256),
        ):
            if value is not None and (
                len(value) != 64 or set(value) - set("0123456789abcdef")
            ):
                raise ValueError(f"{label} must be a SHA-256 hash")

        self.suite = suite
        self.suite_sha256 = canonical_sha256(suite)
        self.task = task
        self.scenario = scenario
        self.contract = contract
        self.scenario_id = scenario_id
        self.control_dt_s = float(control_dt_s)
        self.simulator_revision = simulator_revision
        self.implementation_revision = implementation_revision
        self.output_dir = Path(output_dir)
        self.capture_initial_snapshot = bool(capture_initial_snapshot)
        self.capture_failure_snapshot = bool(capture_failure_snapshot)
        self.start_snapshot_sha256 = start_snapshot_sha256
        self.restore_audit_sha256 = restore_audit_sha256
        self.env_id = int(env_id)
        self.detector = SemanticEventDetector(
            scenario["event_detector"],
            control_dt_s=self.control_dt_s,
            env_id=self.env_id,
        )
        self.trace_path = self.output_dir / "runtime-trace.jsonl"
        self.episode_seed: int | None = None
        self.control_step = 0
        self.trigger_step: int | None = None
        self.trigger_observation: EventObservation | None = None
        self.trigger_signals: dict[str, Any] = {}
        self.action_window: tuple[str, dict[str, Any]] | None = None
        self.action_samples_modified = 0
        self.failure_snapshot_due_step: int | None = None
        self.initial_snapshot_sha256: str | None = None
        self.failure_snapshot_sha256: str | None = None

    def _require_fresh_sidecars(self) -> None:
        stale = [
            path
            for path in (
                self.trace_path,
                self.output_dir / "runtime-summary.json",
                self.output_dir / "runtime-summary.json.tmp",
                self.output_dir / "injector-audit.jsonl",
            )
            if path.exists()
        ]
        if stale:
            joined = ", ".join(str(path) for path in stale)
            raise RuntimeError(
                "recovery output directory contains stale mutable sidecars; "
                f"use a fresh run directory: {joined}"
            )

    @property
    def triggered(self) -> bool:
        return self.trigger_step is not None

    @property
    def failure_capture_complete(self) -> bool:
        return self.failure_snapshot_sha256 is not None

    def _snapshot_assets(self, env: Any, *, failure: bool) -> tuple[str, ...]:
        names = list(TASK_MUTABLE_ASSETS[self.task["id"]])
        missing = [name for name in names if name not in env.scene.keys()]
        if missing:
            raise ValueError(f"snapshot scene assets are missing: {missing}")
        return tuple(names)

    def _capture(self, env: Any, *, failure: bool) -> str:
        assert self.episode_seed is not None
        label = self.scenario_id if failure else "initial"
        snapshot = capture_scene_snapshot(
            env,
            task_id=self.task["id"],
            event_id=label,
            episode_seed=self.episode_seed,
            scene_assets=self._snapshot_assets(env, failure=failure),
            simulator_revision=self.simulator_revision,
            env_id=self.env_id,
        )
        path = self.output_dir / "snapshots" / self.task["id"] / f"{label}.json"
        digest = _write_snapshot_once(path, snapshot)
        _append_jsonl(
            self.trace_path,
            {
                "event": "failure_snapshot_captured" if failure else "initial_snapshot_captured",
                "scenario_id": self.scenario_id,
                "episode_seed": self.episode_seed,
                "control_step": self.control_step,
                "snapshot_path": str(path),
                "snapshot_sha256": digest,
            },
        )
        return digest

    def reset(self, env: Any, *, episode_seed: int) -> None:
        self._require_fresh_sidecars()
        reset_one_shot_injectors(env)
        self.detector.reset(env)
        self.episode_seed = int(episode_seed)
        self.control_step = 0
        self.trigger_step = None
        self.trigger_observation = None
        self.trigger_signals = {}
        self.action_window = None
        self.action_samples_modified = 0
        self.failure_snapshot_due_step = None
        self.initial_snapshot_sha256 = None
        self.failure_snapshot_sha256 = None
        _append_jsonl(
            self.trace_path,
            {
                "event": "episode_reset",
                "suite_sha256": self.suite_sha256,
                "task_id": self.task["id"],
                "scenario_id": self.scenario_id,
                "episode_seed": self.episode_seed,
                "control_dt_s": self.control_dt_s,
                "environment_index": self.env_id,
                "simulator_revision": self.simulator_revision,
                "implementation_revision": self.implementation_revision,
                "start_snapshot_sha256": self.start_snapshot_sha256,
                "restore_audit_sha256": self.restore_audit_sha256,
            },
        )
        if self.capture_initial_snapshot:
            locked_seed = int(self.suite["admission_capture"]["snapshot_seed"])
            if self.episode_seed != locked_seed:
                raise ValueError(
                    f"initial snapshot seed is {self.episode_seed}, expected {locked_seed}"
                )
            self.initial_snapshot_sha256 = self._capture(env, failure=False)

    def _activate_action_window(self, injector_id: str, parameters: dict[str, Any]) -> None:
        if injector_id == "release-grasp-contact":
            self.action_window = (
                "release-grasp-contact",
                {"duration_s": float(parameters["duration_s"])},
            )
        elif injector_id == "attenuate-sonic-latent":
            self.action_window = (
                "attenuate-sonic-latent",
                {
                    "duration_s": float(parameters["duration_s"]),
                    "scale": float(parameters["scale"]),
                },
            )

    def _inject(self, env: Any, signals: dict[str, Any]) -> None:
        assert self.episode_seed is not None
        injector = self.scenario["injector"]
        injector_id = injector["id"]
        parameters = injector["parameters"]
        audit_path = str(self.output_dir / "injector-audit.jsonl")

        if injector_id == "release-grasp-contact":
            self._activate_action_window(injector_id, parameters)
        elif injector_id == "drop-object-and-root-velocity":
            if str(parameters["object_asset_name"]) not in env.scene.keys():
                raise ValueError("drop-object target asset is absent from the live scene")
            self._activate_action_window(
                "release-grasp-contact",
                {"duration_s": parameters["release_duration_s"]},
            )
            apply_root_local_lateral_velocity_once(
                env,
                None,
                injector_id=injector_id,
                lateral_mps=float(parameters["lateral_mps"]),
                lateral_direction_robot=str(parameters["lateral_direction_robot"]),
                audit_path=audit_path,
                episode_seed=self.episode_seed,
            )
        elif injector_id == "shift-door-assembly-handle-frame":
            distance = float(parameters["translation_m"])
            axis = str(parameters["translation_axis_door_local"])
            translation = (distance, 0.0, 0.0) if axis == "x" else (0.0, distance, 0.0)
            apply_asset_local_translation_once(
                env,
                None,
                injector_id=injector_id,
                asset_name=str(parameters["door_asset_name"]),
                local_translation_m=translation,
                audit_path=audit_path,
                episode_seed=self.episode_seed,
            )
        elif injector_id == "place-doorway-obstacle":
            place_asset_relative_once(
                env,
                None,
                injector_id=injector_id,
                asset_name="hrvla_doorway_obstacle",
                reference_asset_name="door",
                reference_local_position_m=tuple(parameters["door_local_center_m"]),
                audit_path=audit_path,
                episode_seed=self.episode_seed,
            )
        elif injector_id == "attenuate-sonic-latent":
            self._activate_action_window(injector_id, parameters)
        elif injector_id == "root-lateral-velocity-delta":
            apply_root_local_lateral_velocity_once(
                env,
                None,
                injector_id=injector_id,
                lateral_mps=float(parameters["lateral_mps"]),
                lateral_direction_robot=str(parameters["lateral_direction_robot"]),
                audit_path=audit_path,
                episode_seed=self.episode_seed,
            )
        elif injector_id == "support-foot-lateral-impulse":
            direction = local_lateral_vector_world(
                env,
                magnitude=float(parameters["impulse_ns"]),
                lateral_direction_robot=str(parameters["lateral_direction_robot"]),
                env_id=self.env_id,
            )
            apply_body_impulse_once(
                env,
                None,
                injector_id=injector_id,
                asset_name="robot",
                body_name=str(signals["support_foot_body_name"]),
                impulse_world_ns=direction,
                control_dt_s=self.control_dt_s,
                audit_path=audit_path,
                episode_seed=self.episode_seed,
            )
        elif injector_id == "upper-body-contact-impulse":
            unit = [float(value) for value in signals["recoil_direction_world"]]
            impulse = tuple(float(parameters["impulse_ns"]) * value for value in unit)
            apply_body_impulse_once(
                env,
                None,
                injector_id=injector_id,
                asset_name="robot",
                body_name=str(parameters["body_name"]),
                impulse_world_ns=impulse,
                control_dt_s=self.control_dt_s,
                audit_path=audit_path,
                episode_seed=self.episode_seed,
            )
        elif injector_id == "place-path-obstacle":
            place_asset_relative_once(
                env,
                None,
                injector_id=injector_id,
                asset_name=str(parameters["asset_name"]),
                reference_asset_name="robot",
                reference_local_position_m=(
                    float(parameters["distance_ahead_m"]),
                    float(parameters["clearance_m"]),
                    0.0,
                ),
                preserve_height=True,
                audit_path=audit_path,
                episode_seed=self.episode_seed,
            )
        else:
            raise ValueError(f"unsupported runtime injector: {injector_id}")

        if self.capture_failure_snapshot:
            settle_s = float(self.scenario["failure_snapshot_settle_s"])
            self.failure_snapshot_due_step = self.control_step + math.ceil(
                settle_s / self.control_dt_s
            )

    def _record_trigger(
        self,
        env: Any,
        observation: EventObservation,
        signals: dict[str, Any],
    ) -> None:
        if self.triggered:
            return
        self.trigger_step = self.control_step
        self.trigger_observation = observation
        self.trigger_signals = signals
        self._inject(env, signals)
        _append_jsonl(
            self.trace_path,
            {
                "event": "semantic_boundary_triggered",
                "scenario_id": self.scenario_id,
                "episode_seed": self.episode_seed,
                "control_step": self.control_step,
                "detector_id": observation.detector_id,
                "detector_sample_index": observation.sample_index,
                "diagnostics": observation.diagnostics,
                "signals": signals,
                "injector_id": self.scenario["injector"]["id"],
            },
        )

    def before_control_step(self, env: Any, *, task_success: bool) -> None:
        if self.episode_seed is None:
            raise RuntimeError("runtime.reset is required before stepping")
        self.control_step += 1
        clear_expired_body_impulses(env)
        if (
            self.failure_snapshot_due_step is not None
            and self.failure_snapshot_sha256 is None
            and self.control_step >= self.failure_snapshot_due_step
        ):
            self.failure_snapshot_sha256 = self._capture(env, failure=True)

        detector_id = self.scenario["event_detector"]["id"]
        if self.triggered or detector_id == "first-hand-close":
            return
        parameters = self.scenario["event_detector"]["parameters"]
        signals = signals_for_detector(detector_id, env, parameters)
        observation = self.detector.observe(
            env, task_success=bool(task_success), signals=signals
        )
        if observation.triggered:
            self._record_trigger(env, observation, signals)

    def transform_vla_action(
        self,
        env: Any,
        action: Any,
        *,
        task_success: bool,
    ) -> Any:
        if self.episode_seed is None:
            raise RuntimeError("runtime.reset is required before transforming actions")
        if self.scenario["event_detector"]["id"] == "first-hand-close" and not self.triggered:
            observation = self.detector.observe(
                env,
                task_success=bool(task_success),
                semantic_action=action,
            )
            if observation.triggered:
                self._record_trigger(env, observation, {})
        if not self.triggered or self.action_window is None:
            return action
        injector_id, parameters = self.action_window
        assert self.trigger_step is not None
        elapsed_s = max(0.0, (self.control_step - self.trigger_step) * self.control_dt_s)
        output = apply_action_window(
            action,
            injector_id=injector_id,
            elapsed_s=elapsed_s,
            parameters=parameters,
        )
        if elapsed_s < float(parameters["duration_s"]):
            import numpy as np

            before = np.asarray(action, dtype=np.float32)
            after = np.asarray(output, dtype=np.float32)
            changed = not np.array_equal(before, after)
            if changed:
                self.action_samples_modified += 1
            _append_jsonl(
                self.trace_path,
                {
                    "event": "action_window_applied",
                    "scenario_id": self.scenario_id,
                    "episode_seed": self.episode_seed,
                    "control_step": self.control_step,
                    "injector_id": injector_id,
                    "elapsed_s": elapsed_s,
                    "shape": list(before.shape),
                    "input_sha256": canonical_sha256(before.tolist()),
                    "output_sha256": canonical_sha256(after.tolist()),
                    "changed": changed,
                    "delta_l2": float(np.linalg.norm(after - before)),
                },
            )
        return output

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "suite_sha256": self.suite_sha256,
            "task_id": self.task["id"],
            "scenario_id": self.scenario_id,
            "episode_seed": self.episode_seed,
            "implementation_revision": self.implementation_revision,
            "triggered": self.triggered,
            "trigger_control_step": self.trigger_step,
            "detector_id": self.scenario["event_detector"]["id"],
            "injector_id": self.scenario["injector"]["id"],
            "action_samples_modified": self.action_samples_modified,
            "initial_snapshot_sha256": self.initial_snapshot_sha256,
            "start_snapshot_sha256": self.start_snapshot_sha256,
            "restore_audit_sha256": self.restore_audit_sha256,
            "failure_snapshot_sha256": self.failure_snapshot_sha256,
            "failure_capture_complete": self.failure_capture_complete,
            "runtime_validated": False,
            "claim_boundary": "runtime_validated remains false until an Isaac Sim trace audit passes",
        }


__all__ = [
    "HumanoidArenaRecoveryRuntime",
    "TASK_MUTABLE_ASSETS",
    "configure_recovery_scene",
]
