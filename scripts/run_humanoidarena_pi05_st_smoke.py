#!/usr/bin/env python3
"""Run one bounded PI0.5 ST/SONIC Isaac integration episode with GPU INT8.

This development smoke run checks the live HTTP prompt hook and action40
contract. It is deliberately not a benchmark-complete or paired success claim.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/run_humanoidarena_pi05_st_episode.py"


def _baseline(runtime_root: Path):
    path = runtime_root / "scripts/run_humanoidarena_baseline_matrix_fast.py"
    spec = importlib.util.spec_from_file_location("hrvla_pinned_pi05_baseline_driver", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pinned PI0.5 baseline driver: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_sim_command(
    baseline, *, runtime_root: Path, task_key: str, mode: str,
    output: Path, port: int, seed: int, max_steps: int,
) -> list[str]:
    command = baseline._sim_command(
        task_name=task_key, mode=mode, batch_path=output / "unused-batch.json",
        port=port, record_video_every_n=1, step_log_every_n=0,
    )
    command[2] = str(WRAPPER)
    batch_position = command.index("--episode_batch_json")
    del command[batch_position : batch_position + 2]
    command[command.index("--max_steps") + 1] = str(max_steps)
    command[command.index("--seed") + 1] = str(seed)
    episode_seed = baseline._derive_episode_seed(
        baseline.TASKS[task_key]["task_id"], seed, 0
    )
    command.extend([
        "--episode_seed", str(episode_seed),
        "--episode_index", "0",
        "--result_json", str(output / "episode.json"),
        "--success_video_dir", str(output / "videos/success"),
        "--failure_video_dir", str(output / "videos/failure"),
        "--model_label", "pi05_st_integration_smoke",
        "--eval_model_path", str(
            runtime_root / "_artifacts/HumanoidArena/models" / baseline.TASKS[task_key]["model"]
        ),
    ])
    return command[:3] + [
        "--runtime-root", str(runtime_root),
        "--method-output-dir", str(output / "method"),
    ] + command[3:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--task-key", default="pp_box")
    parser.add_argument("--mode", default="base_test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    runtime_root = args.runtime_root.resolve(strict=True)
    baseline = _baseline(runtime_root)
    if args.task_key not in baseline.TASKS or args.mode not in baseline.MODES:
        parser.error("task-key or mode is outside the pinned PI0.5 matrix")
    if args.max_steps < 1 or args.seed < 0 or args.port not in range(1024, 65536):
        parser.error("max-steps, seed, or port is invalid")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite smoke output: {output}")
    model_root = runtime_root / "_artifacts/HumanoidArena/models"
    baseline._validate_runtime(model_root)
    model = model_root / baseline.TASKS[args.task_key]["model"]
    server_command = baseline._server_command(
        model, args.port, baseline.CUDA_INT8_BACKEND
    )
    sim_command = build_sim_command(
        baseline, runtime_root=runtime_root, task_key=args.task_key,
        mode=args.mode, output=output, port=args.port, seed=args.seed,
        max_steps=args.max_steps,
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
            server_command, cwd=runtime_root, env=baseline._server_env(
                16, 2, 4, baseline.CUDA_INT8_BACKEND
            ), stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            baseline._wait_for_server(args.port, 600.0)
            baseline._warm_cuda_int8_policy(args.port, args.task_key)
            if baseline._gpu_used_mib() > 8500:
                raise RuntimeError("warm INT8 policy exceeded the pinned 8500 MiB admission limit")
            with (output / "sim.log").open("w", encoding="utf-8", buffering=1) as sim_log:
                result = subprocess.run(
                    sim_command, cwd=baseline.ISAACLAB_ROOT, env=baseline._sim_env(),
                    stdout=sim_log, stderr=subprocess.STDOUT, check=False,
                )
            if result.returncode != 0:
                raise RuntimeError(f"Isaac smoke failed with code {result.returncode}; see {output / 'sim.log'}")
        finally:
            baseline._terminate_group(server)
    episode = json.loads((output / "episode.json").read_text(encoding="utf-8"))
    trace_path = output / "method/method-trace.jsonl"
    traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    decisions = [row for row in traces if row.get("event") == "instruction_selected"]
    if (not decisions or any(row.get("policy_action_dim") != 40 for row in decisions)
            or episode.get("hrvla_method", {}).get("policy_requests") != len(decisions)):
        raise RuntimeError("PI0.5 ST smoke lacks an audited action40 prompt request")
    print(json.dumps({
        "result_json": str(output / "episode.json"),
        "success": episode.get("success"),
        "failure_reason": episode.get("failure_reason"),
        "policy_requests": len(decisions),
        "skills": [row.get("selected_skill_id") for row in decisions],
        "video_dirs": [str(output / "videos/success"), str(output / "videos/failure")],
        "claim_boundary": "single development smoke; no paired benchmark claim",
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
