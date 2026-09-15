#!/usr/bin/env python3
"""Run a bounded HumanoidArena task smoke test and record machine-readable evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task-id", required=True)
parser.add_argument("--env-config", required=True)
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--isaaclab-root", type=Path, required=True)
parser.add_argument("--evidence-output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

source_root = args.source_root.resolve()
isaaclab_root = args.isaaclab_root.resolve()
project_root = source_root / "isaaclab_twist2_g1"
os.environ["PROJECT_ROOT"] = str(project_root)
sys.path.insert(0, str(project_root))

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import tasks  # noqa: E402, F401
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from tasks.common_env_config import apply_env_config_yaml  # noqa: E402


def main() -> None:
    started = time.monotonic()
    env = None
    evidence = {
        "schema_version": 1,
        "benchmark": "HumanoidArena",
        "task_id": args.task_id,
        "env_config": args.env_config,
        "seed": args.seed,
        "requested_steps": args.steps,
        "source_revision": _git_revision(source_root),
        "isaaclab_revision": _git_revision(isaaclab_root),
        "isaac_sim_version": "5.0.0",
        "device": args.device,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "fail",
    }
    try:
        cfg = parse_env_cfg(args.task_id, device=args.device, num_envs=1)
        resolved_config = apply_env_config_yaml(
            cfg,
            args.env_config,
            task_name=args.task_id,
            route_name="sonic",
        )
        env = gym.make(args.task_id, cfg=cfg)
        env.reset(seed=args.seed)
        action_shape = tuple(env.action_space.shape)
        action = torch.zeros(action_shape, device=env.unwrapped.device)
        for _ in range(args.steps):
            env.step(action)
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        evidence.update(
            {
                "status": "pass",
                "completed_steps": args.steps,
                "action_shape": list(action_shape),
                "resolved_env_config": str(resolved_config),
                "runtime_device": str(env.unwrapped.device),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        evidence["elapsed_seconds"] = time.monotonic() - started
        evidence["finished_at"] = datetime.now(timezone.utc).isoformat()
        args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_output.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
