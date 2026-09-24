#!/usr/bin/env python3
"""Run one live PI0.5-STR recovery episode through the pinned HA stack."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import socket
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/run_humanoidarena_internal_episode.py"
DEFAULT_SUITE = ROOT / "benchmark/suites/hrvla_recovery_v0.json"


def _baseline(runtime_root: Path):
    path = runtime_root / "scripts/run_humanoidarena_baseline_matrix_fast.py"
    spec = importlib.util.spec_from_file_location("hrvla_pinned_pi05_str_driver", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pinned PI0.5 baseline driver: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scenario(suite: dict, scenario_id: str) -> tuple[dict, dict]:
    for task in suite["tasks"]:
        for scenario in task["scenarios"]:
            if scenario["id"] == scenario_id:
                return task, scenario
    raise ValueError(f"unknown recovery scenario: {scenario_id}")


def _start_snapshot(
    capture_manifest: dict, scenario: dict, capture_root: Path,
) -> tuple[Path, str]:
    capture = next(
        (row for row in capture_manifest["scenarios"]
         if row["scenario_id"] == scenario["id"]),
        None,
    )
    if capture is None:
        raise ValueError(f"capture manifest has no {scenario['id']} row")
    prefix = "failure" if scenario["protocol"] == "failure_start" else "initial"
    relative = capture.get(f"{prefix}_snapshot_path")
    expected = capture.get(f"{prefix}_snapshot_sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise ValueError(f"capture manifest lacks the {prefix} snapshot")
    path = (capture_root / relative).resolve(strict=True)
    if not path.is_relative_to(capture_root.resolve()):
        raise ValueError("snapshot path escapes the capture root")
    return path, expected


def build_sim_command(
    baseline, *, runtime_root: Path, suite_path: Path, scenario: dict,
    task_key: str, output: Path, snapshot_path: Path, snapshot_sha256: str,
    port: int, seed: int, max_steps: int,
) -> list[str]:
    command = baseline._sim_command(
        task_name=task_key, mode="base_test",
        batch_path=output / "unused-batch.json", port=port,
        record_video_every_n=1, step_log_every_n=0,
    )
    evaluator_index = command.index(str(baseline.SIM_EVALUATOR))
    command[evaluator_index] = str(WRAPPER)
    batch_position = command.index("--episode_batch_json")
    del command[batch_position : batch_position + 2]
    command[command.index("--max_steps") + 1] = str(max_steps)
    command[command.index("--seed") + 1] = str(seed)
    episode_seed = baseline._derive_episode_seed(
        baseline.TASKS[task_key]["task_id"], seed, 0
    )
    wrapper_args = [
        "--runtime-root", str(runtime_root),
        "--internal-method", "pi05_str",
        "--suite", str(suite_path),
        "--recovery-scenario", scenario["id"],
        "--internal-output-dir", str(output / "method"),
        "--start-snapshot", str(snapshot_path),
        "--expected-start-snapshot-sha256", snapshot_sha256,
    ]
    command[evaluator_index + 1:evaluator_index + 1] = wrapper_args
    command.extend([
        "--episode_seed", str(episode_seed),
        "--episode_index", "0",
        "--result_json", str(output / "episode.json"),
        "--success_video_dir", str(output / "videos/success"),
        "--failure_video_dir", str(output / "videos/failure"),
        "--model_label", "pi05_str_integration_smoke",
        "--eval_model_path", str(
            runtime_root / "_artifacts/HumanoidArena/models"
            / baseline.TASKS[task_key]["model"]
        ),
    ])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--scenario", default="box-missed-grasp-retry")
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    runtime_root = args.runtime_root.resolve(strict=True)
    suite_path = args.suite.resolve(strict=True)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    task, scenario = _scenario(suite, args.scenario)
    capture_manifest = json.loads(
        args.capture_manifest.resolve(strict=True).read_text(encoding="utf-8")
    )
    capture_root = args.capture_root.resolve(strict=True)
    snapshot_path, snapshot_sha256 = _start_snapshot(
        capture_manifest, scenario, capture_root
    )
    baseline = _baseline(runtime_root)
    task_key = task["humanoidarena_task_key"]
    if task_key not in baseline.TASKS:
        parser.error("scenario task is outside the pinned PI0.5 matrix")
    max_steps = args.max_steps or int(baseline.TASKS[task_key]["max_steps"])
    if max_steps < 1 or args.seed < 0 or args.port not in range(1024, 65536):
        parser.error("max-steps, seed, or port is invalid")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite smoke output: {output}")
    model_root = runtime_root / "_artifacts/HumanoidArena/models"
    baseline._validate_runtime(model_root)
    model = model_root / baseline.TASKS[task_key]["model"]
    server_command = baseline._server_command(
        model, args.port, baseline.CUDA_INT8_BACKEND
    )
    sim_command = build_sim_command(
        baseline, runtime_root=runtime_root, suite_path=suite_path,
        scenario=scenario, task_key=task_key, output=output,
        snapshot_path=snapshot_path, snapshot_sha256=snapshot_sha256,
        port=args.port, seed=args.seed, max_steps=max_steps,
    )
    print(json.dumps({"server": server_command, "sim": sim_command}, indent=2), flush=True)
    if args.dry_run:
        return 0
    if baseline._gpu_used_mib() > 1024:
        raise RuntimeError("GPU is not idle enough for PI0.5+Isaac coexistence")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError(f"policy port {args.port} is already occupied")
    output.mkdir(parents=True)
    with (output / "server.log").open("w", encoding="utf-8", buffering=1) as server_log:
        server = subprocess.Popen(
            server_command, cwd=runtime_root,
            env=baseline._server_env(16, 2, 4, baseline.CUDA_INT8_BACKEND),
            stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            baseline._wait_for_server(args.port, 600.0)
            baseline._warm_cuda_int8_policy(args.port, task_key)
            if baseline._gpu_used_mib() > 8500:
                raise RuntimeError("warm INT8 policy exceeded the pinned 8500 MiB limit")
            with (output / "sim.log").open("w", encoding="utf-8", buffering=1) as sim_log:
                result = subprocess.run(
                    sim_command, cwd=baseline.ISAACLAB_ROOT, env=baseline._sim_env(),
                    stdout=sim_log, stderr=subprocess.STDOUT, check=False,
                )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Isaac smoke failed with code {result.returncode}; see {output / 'sim.log'}"
                )
            if not (output / "episode.json").is_file():
                raise RuntimeError("Isaac exited without an episode result")
        finally:
            baseline._terminate_group(server)
    episode = json.loads((output / "episode.json").read_text(encoding="utf-8"))
    summary = episode.get("hrvla_method", {})
    recovery = episode.get("hrvla_recovery", {})
    if (summary.get("method_id") != "pi05_str"
            or int(summary.get("recovery_decisions", 0)) < 1
            or summary.get("first_recovery_decision_control_step") is None
            or recovery.get("triggered") is not True):
        raise RuntimeError("live smoke did not exercise PI0.5-STR recovery")
    print(json.dumps({
        "result_json": str(output / "episode.json"),
        "scenario_id": scenario["id"],
        "success": episode.get("success"),
        "failure_reason": episode.get("failure_reason"),
        "policy_requests": summary.get("policy_requests"),
        "recovery_decisions": summary.get("recovery_decisions"),
        "recovery_triggered": recovery.get("triggered"),
        "claim_boundary": "single development smoke; no paired benchmark claim",
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
