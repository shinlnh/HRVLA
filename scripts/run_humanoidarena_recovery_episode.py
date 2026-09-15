#!/usr/bin/env python3
"""Run one upstream HumanoidArena episode with an audited recovery scenario."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.humanoidarena_recovery_runtime import (  # noqa: E402
    HumanoidArenaRecoveryRuntime,
    configure_recovery_scene,
)
from hrvla_bench.plan import canonical_sha256, load_json  # noqa: E402
from hrvla_bench.isaac_snapshot import load_snapshot  # noqa: E402
from hrvla_bench.recovery_restore import restore_snapshot_for_trial  # noqa: E402


EVALUATOR = (
    ROOT
    / "_vendor/HumanoidArena/isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/sim_eval_vla.py"
)
ISAACLAB_REVISION = "46dff135f44683f031edf346e544fcfd8456b2bb"
TASK_IDS = {
    "pick_and_place_box": "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby",
    "open_door": "Isaac-Move-Open-Door-G129-Dex3-Wholebody",
    "double_desk": "Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody",
    "football": "Isaac-Move-Football-Single-G129-Dex3-Wholebody",
    "sit_sofa": "Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody",
    "boxing": "Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody",
    "visual_navigation": "Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody",
}


class _EncoderProxy:
    def __init__(self, encoder: Any, runtime_state: dict[str, Any]) -> None:
        self._encoder = encoder
        self._runtime_state = runtime_state

    def __getattr__(self, name: str) -> Any:
        return getattr(self._encoder, name)

    def run(self, *args, **kwargs):
        outputs = list(self._encoder.run(*args, **kwargs))
        runtime = self._runtime_state.get("runtime")
        env = self._runtime_state.get("env")
        if runtime is not None and env is not None:
            outputs[0] = runtime.transform_vla_action(
                env,
                outputs[0],
                task_success=bool(self._runtime_state.get("task_success", False)),
            )
        return outputs


def _single_episode_contract(remaining: list[str]) -> tuple[str, int]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode_seed", type=int)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episode_batch_json", default="")
    values, _ = parser.parse_known_args(remaining)
    if values.episode_batch_json:
        payload = json.loads(Path(values.episode_batch_json).read_text(encoding="utf-8"))
        episodes = payload.get("episodes", [])
        if not isinstance(episodes, list) or len(episodes) != 1:
            raise ValueError("recovery runtime requires exactly one episode per process")
        episode_seed = int(episodes[0]["episode_seed"])
    else:
        episode_seed = int(values.episode_seed if values.episode_seed is not None else values.seed)
    return values.task, episode_seed


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("hrvla_upstream_sim_eval_vla", EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load upstream evaluator: {EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_runtime_hooks(
    module: Any,
    suite: dict[str, Any],
    scenario_id: str,
    task_id: str,
    output_dir: Path,
    *,
    capture_initial_snapshot: bool,
    capture_failure_snapshot: bool,
    start_snapshot_path: Path | None = None,
    expected_start_snapshot_sha256: str | None = None,
    restore_only: bool = False,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "runtime": None,
        "env": None,
        "task_success": False,
        "inside_episode": False,
        "provider_hooked": False,
        "restore_audit": None,
    }

    import gymnasium as gym

    original_make = gym.make

    def recovery_make(env_id, *args, **kwargs):
        configure_recovery_scene(kwargs["cfg"], suite, task_id)
        return original_make(env_id, *args, **kwargs)

    gym.make = recovery_make

    original_run_episode = module._run_episode_once

    def recovery_run_episode(*args, **kwargs):
        state["inside_episode"] = True
        try:
            return original_run_episode(*args, **kwargs)
        finally:
            state["inside_episode"] = False

    module._run_episode_once = recovery_run_episode

    original_reset = module._reset_environment_for_episode

    start_snapshot = (
        None if start_snapshot_path is None else load_snapshot(start_snapshot_path.resolve())
    )

    def recovery_reset(env, env_cfg, episode_seed):
        environment_seed = (
            int(episode_seed)
            if start_snapshot is None
            else int(start_snapshot["episode_seed"])
        )
        result = original_reset(env, env_cfg, environment_seed)
        if not state["inside_episode"]:
            return result
        restore_audit = None
        if start_snapshot is not None:
            expected_event = scenario_id if restore_only else "initial"
            assert start_snapshot_path is not None
            assert expected_start_snapshot_sha256 is not None
            restore_audit = restore_snapshot_for_trial(
                env,
                start_snapshot_path,
                expected_snapshot_sha256=expected_start_snapshot_sha256,
                expected_task_id=task_id,
                expected_event_id=expected_event,
                expected_simulator_revision=ISAACLAB_REVISION,
                policy_rollout_seed=int(episode_seed),
                audit_path=output_dir / "restore-audit.json",
            )
        if restore_only:
            state.update(
                runtime=None,
                env=env,
                task_success=False,
                provider_hooked=True,
                restore_audit=restore_audit,
            )
            return result
        control_dt = getattr(env, "step_dt", None)
        if control_dt is None:
            control_dt = float(env.physics_dt) * int(env_cfg.decimation)
        runtime = HumanoidArenaRecoveryRuntime(
            suite,
            scenario_id,
            control_dt_s=float(control_dt),
            simulator_revision=ISAACLAB_REVISION,
            output_dir=output_dir,
            capture_initial_snapshot=capture_initial_snapshot,
            capture_failure_snapshot=capture_failure_snapshot,
            start_snapshot_sha256=(
                None if restore_audit is None else restore_audit["snapshot_sha256"]
            ),
            restore_audit_sha256=(
                None if restore_audit is None else restore_audit["audit_sha256"]
            ),
        )
        runtime.reset(env, episode_seed=int(episode_seed))
        state.update(
            runtime=runtime,
            env=env,
            task_success=False,
            provider_hooked=False,
            restore_audit=restore_audit,
        )
        return result

    module._reset_environment_for_episode = recovery_reset

    original_reward = module._extract_reward_info

    def recovery_reward(env):
        result = original_reward(env)
        state["task_success"] = float(result["raw_total"]) >= 1.0
        return result

    module._extract_reward_info = recovery_reward

    original_notify = module._notify_action_provider_env_reset

    def recovery_notify(provider):
        result = original_notify(provider)
        runtime = state.get("runtime")
        env = state.get("env")
        if env is None:
            raise RuntimeError("recovery runtime was not initialized at episode reset")
        if restore_only:
            return result
        if runtime is None:
            raise RuntimeError("recovery runtime was not initialized at episode reset")
        if state["provider_hooked"]:
            return result

        original_get_action = provider.get_action

        def recovery_get_action(current_env):
            runtime.before_control_step(
                current_env, task_success=bool(state["task_success"])
            )
            return original_get_action(current_env)

        provider.get_action = recovery_get_action

        seam = runtime.contract["interface_seam"]
        if seam in {"semantic-action40", "scene+semantic-action40"}:
            original_pop_action = provider._pop_lerobot_action

            def recovery_pop_action():
                action = original_pop_action()
                return runtime.transform_vla_action(
                    env,
                    action,
                    task_success=bool(state["task_success"]),
                )

            provider._pop_lerobot_action = recovery_pop_action
        elif seam == "post-encoder-latent64":
            if provider._encoder is None:
                raise RuntimeError("SONIC encoder is unavailable for latent64 perturbation")
            provider._encoder = _EncoderProxy(provider._encoder, state)
        state["provider_hooked"] = True
        return result

    module._notify_action_provider_env_reset = recovery_notify

    original_payload = module._build_result_payload

    def recovery_payload(*args, **kwargs):
        payload = original_payload(*args, **kwargs)
        runtime = state.get("runtime")
        restore_audit = state.get("restore_audit")
        if restore_only:
            if restore_audit is None:
                raise RuntimeError("failure-start trial did not produce restore evidence")
            recovery = {
                "schema_version": 1,
                "suite_sha256": canonical_sha256(suite),
                "task_id": task_id,
                "scenario_id": scenario_id,
                "episode_seed": int(payload["episode_seed"]),
                "protocol": "failure_start",
                "failure_injected": False,
                "start_snapshot_sha256": restore_audit["snapshot_sha256"],
                "restore_audit_sha256": restore_audit["audit_sha256"],
                "restore_validated": True,
                "claim_boundary": "snapshot restore evidence; task success remains in the episode result",
            }
        else:
            if runtime is None:
                raise RuntimeError("recovery runtime did not produce an episode summary")
            recovery = runtime.summary()
        payload["hrvla_recovery"] = recovery
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / (
            "trial-summary.json" if restore_only else "runtime-summary.json"
        )
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(recovery, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
        return payload

    module._build_result_payload = recovery_payload
    return state


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--recovery-suite",
        type=Path,
        default=ROOT / "benchmark/suites/hrvla_recovery_v0.json",
    )
    parser.add_argument("--recovery-scenario", required=True)
    parser.add_argument("--recovery-output-dir", type=Path, required=True)
    parser.add_argument("--capture-initial-snapshot", action="store_true")
    parser.add_argument("--capture-failure-snapshot", action="store_true")
    parser.add_argument("--start-snapshot", type=Path)
    parser.add_argument("--expected-start-snapshot-sha256")
    parser.add_argument("--restore-only", action="store_true")
    args, remaining = parser.parse_known_args()
    suite = load_json(args.recovery_suite.resolve())
    upstream_task, episode_seed = _single_episode_contract(remaining)
    task = next(
        (
            task
            for task in suite["tasks"]
            if any(item["id"] == args.recovery_scenario for item in task["scenarios"])
        ),
        None,
    )
    if task is None:
        raise ValueError(f"unknown recovery scenario: {args.recovery_scenario}")
    scenario = next(
        item for item in task["scenarios"] if item["id"] == args.recovery_scenario
    )
    expected_task = TASK_IDS[task["id"]]
    if upstream_task != expected_task:
        raise ValueError(
            f"scenario {args.recovery_scenario} requires task {expected_task}, got {upstream_task}"
        )
    if args.capture_initial_snapshot:
        expected_seed = int(suite["admission_capture"]["snapshot_seed"])
        if episode_seed != expected_seed:
            raise ValueError(
                f"initial snapshot capture requires episode seed {expected_seed}, got {episode_seed}"
            )
    if (args.start_snapshot is None) != (
        args.expected_start_snapshot_sha256 is None
    ):
        raise ValueError("--start-snapshot and --expected-start-snapshot-sha256 are required together")
    if args.start_snapshot is not None and (
        args.capture_initial_snapshot or args.capture_failure_snapshot
    ):
        raise ValueError("snapshot capture and snapshot restore are mutually exclusive")
    if args.restore_only and args.start_snapshot is None:
        raise ValueError("--restore-only requires --start-snapshot")
    if args.restore_only and scenario["protocol"] != "failure_start":
        raise ValueError("--restore-only is valid only for failure_start scenarios")
    if not args.restore_only and args.start_snapshot is not None and scenario["protocol"] != "online_failure":
        raise ValueError("online restore-and-inject mode requires an online_failure scenario")
    if args.start_snapshot is not None:
        snapshot = load_snapshot(args.start_snapshot.resolve())
        expected_event = args.recovery_scenario if args.restore_only else "initial"
        expected_snapshot = {
            "snapshot_sha256": args.expected_start_snapshot_sha256,
            "task_id": task["id"],
            "event_id": expected_event,
            "simulator_revision": ISAACLAB_REVISION,
            "environment_index": 0,
        }
        for key, expected in expected_snapshot.items():
            if snapshot.get(key) != expected:
                raise ValueError(f"start snapshot {key} differs from the trial contract")
        if int(snapshot["episode_seed"]) != int(suite["admission_capture"]["snapshot_seed"]):
            raise ValueError("start snapshot does not use the locked capture seed")
    for key, value in task.get("runtime_environment", {}).items():
        existing = os.environ.get(key)
        if existing is not None and existing != value:
            raise ValueError(f"runtime environment conflict for {key}: {existing!r} != {value!r}")
        os.environ[key] = value

    sys.argv = [sys.argv[0], *remaining]
    module = _load_evaluator()
    _install_runtime_hooks(
        module,
        suite,
        args.recovery_scenario,
        task["id"],
        args.recovery_output_dir.resolve(),
        capture_initial_snapshot=args.capture_initial_snapshot,
        capture_failure_snapshot=args.capture_failure_snapshot,
        start_snapshot_path=(
            None if args.start_snapshot is None else args.start_snapshot.resolve()
        ),
        expected_start_snapshot_sha256=args.expected_start_snapshot_sha256,
        restore_only=args.restore_only,
    )
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
