#!/usr/bin/env python3
"""Run one/batched internal GR00T-ST/STR method through the locked HA evaluator."""

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

from hrvla_bench.humanoidarena_method_hooks import install_method_hooks  # noqa: E402
from hrvla_bench.humanoidarena_method_runtime import load_method_programs  # noqa: E402
from hrvla_bench.isaac_snapshot import load_snapshot  # noqa: E402
from hrvla_bench.plan import load_json  # noqa: E402
from hrvla_bench.recovery_restore import restore_snapshot_for_trial  # noqa: E402


RECOVERY_RUNNER = ROOT / "scripts/run_humanoidarena_recovery_episode.py"
EVALUATOR_RELATIVE = Path(
    "_vendor/HumanoidArena/isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/sim_eval_vla.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECOVERY = _load_module("hrvla_recovery_episode_runner", RECOVERY_RUNNER)


def _install_nominal_restore_hooks(
    module: Any,
    *,
    start_snapshot_path: Path,
    expected_snapshot_sha256: str,
    task_id: str,
    shared_context: dict[str, Any],
) -> dict[str, Any]:
    snapshot = load_snapshot(start_snapshot_path)
    original_reset = module._reset_environment_for_episode
    state: dict[str, Any] = {"restore_audit": None}

    def restore_reset(env, env_cfg, episode_seed):
        result = original_reset(env, env_cfg, int(snapshot["episode_seed"]))
        output = shared_context.get("episode_output_dir")
        if not isinstance(output, Path):
            raise RuntimeError("nominal restore output directory is unresolved")
        state["restore_audit"] = restore_snapshot_for_trial(
            env,
            start_snapshot_path,
            expected_snapshot_sha256=expected_snapshot_sha256,
            expected_task_id=task_id,
            expected_event_id="initial",
            expected_simulator_revision=RECOVERY.ISAACLAB_REVISION,
            policy_rollout_seed=int(episode_seed),
            audit_path=output / "restore-audit.json",
        )
        return result

    module._reset_environment_for_episode = restore_reset
    original_payload = module._build_result_payload

    def restore_payload(*args, **kwargs):
        payload = original_payload(*args, **kwargs)
        audit = state.get("restore_audit")
        if not isinstance(audit, dict):
            raise RuntimeError("nominal trial did not produce restore evidence")
        payload["hrvla_start_restore"] = {
            "snapshot_sha256": audit["snapshot_sha256"],
            "audit_sha256": audit["audit_sha256"],
        }
        return payload

    module._build_result_payload = restore_payload
    return state


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=ROOT,
        help="Runtime checkout containing the pinned HumanoidArena/Isaac assets.",
    )
    parser.add_argument("--internal-method", required=True)
    parser.add_argument(
        "--method-programs",
        type=Path,
        default=ROOT / "benchmark/humanoidarena_method_programs.json",
    )
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--recovery-scenario")
    parser.add_argument("--internal-output-dir", type=Path, required=True)
    parser.add_argument("--start-snapshot", type=Path, required=True)
    parser.add_argument("--expected-start-snapshot-sha256", required=True)
    parser.add_argument("--per-episode-output", action="store_true")
    args, remaining = parser.parse_known_args()

    runtime_root = args.runtime_root.resolve(strict=True)
    evaluator = (runtime_root / EVALUATOR_RELATIVE).resolve(strict=True)
    suite = load_json(args.suite.resolve())
    programs = load_method_programs(args.method_programs.resolve())
    implementation_revision = RECOVERY._repository_revision()
    upstream_task, _episode_seeds = RECOVERY._episode_contract(
        remaining, allow_batch=args.per_episode_output
    )
    reverse_tasks = {value: key for key, value in RECOVERY.TASK_IDS.items()}
    if upstream_task not in reverse_tasks:
        raise ValueError(f"upstream task is outside the internal suite: {upstream_task}")
    task_id = reverse_tasks[upstream_task]
    task = next(row for row in suite["tasks"] if row["id"] == task_id)
    scenario = None
    if args.recovery_scenario:
        scenario = next(
            (row for row in task["scenarios"] if row["id"] == args.recovery_scenario),
            None,
        )
        if scenario is None:
            raise ValueError("recovery scenario does not belong to the upstream task")
    snapshot = load_snapshot(args.start_snapshot.resolve())
    expected_event = scenario["id"] if scenario and scenario["protocol"] == "failure_start" else "initial"
    expected_snapshot = {
        "snapshot_sha256": args.expected_start_snapshot_sha256,
        "task_id": task_id,
        "event_id": expected_event,
        "simulator_revision": RECOVERY.ISAACLAB_REVISION,
        "environment_index": 0,
    }
    for key, expected in expected_snapshot.items():
        if snapshot.get(key) != expected:
            raise ValueError(f"start snapshot {key} differs from the internal trial contract")
    for key, value in task.get("runtime_environment", {}).items():
        existing = os.environ.get(key)
        if existing is not None and existing != value:
            raise ValueError(f"runtime environment conflict for {key}: {existing!r} != {value!r}")
        os.environ[key] = value

    module = _load_module("hrvla_internal_upstream_evaluator", evaluator)
    shared_context: dict[str, Any] = {}
    recovery_state = None
    failure_start = bool(scenario and scenario["protocol"] == "failure_start")
    if scenario is None:
        _install_nominal_restore_hooks(
            module,
            start_snapshot_path=args.start_snapshot.resolve(),
            expected_snapshot_sha256=args.expected_start_snapshot_sha256,
            task_id=task_id,
            shared_context=shared_context,
        )
    else:
        recovery_state = RECOVERY._install_runtime_hooks(
            module,
            suite,
            scenario["id"],
            task_id,
            args.internal_output_dir.resolve(),
            capture_initial_snapshot=False,
            capture_failure_snapshot=False,
            start_snapshot_path=args.start_snapshot.resolve(),
            expected_start_snapshot_sha256=args.expected_start_snapshot_sha256,
            restore_only=failure_start,
            per_episode_output=args.per_episode_output,
            implementation_revision=implementation_revision,
        )
    install_method_hooks(
        module,
        programs,
        method_id=args.internal_method,
        task_id=task_id,
        output_dir=args.internal_output_dir.resolve(),
        implementation_revision=implementation_revision,
        recovery_state=recovery_state,
        recovery_scenario_id=None if scenario is None else scenario["id"],
        failure_start=failure_start,
        per_episode_output=args.per_episode_output,
        shared_context=shared_context,
    )
    sys.argv = [sys.argv[0], *remaining]
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
