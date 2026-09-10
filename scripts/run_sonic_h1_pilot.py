#!/usr/bin/env python3
"""Run the audited one-shot H1 lateral-push SONIC diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "_artifacts" / "sonic_eval" / "h1_push_035"


def hydra_overrides(lateral_mps: float, audit_path: Path) -> list[str]:
    velocity = {
        "x": 0.0,
        "y": lateral_mps,
        "z": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    }
    overrides = [
        "++manager_env.config.train_only_events=[]",
        (
            "++manager_env.events.push_robot.func="
            "hrvla_bench.isaac_events:push_once_by_setting_velocity"
        ),
        "++manager_env.events.push_robot.mode=interval",
        "++manager_env.events.push_robot.interval_range_s=[2.0,2.0]",
    ]
    overrides.extend(
        f"++manager_env.events.push_robot.params.velocity_range.{axis}=[{value:g},{value:g}]"
        for axis, value in velocity.items()
    )
    overrides.append(
        "++manager_env.events.push_robot.params.audit_path=" + str(audit_path.resolve())
    )
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime", choices=("auto", "pip", "workstation"), default="auto"
    )
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--lateral-mps", type=float, default=0.35)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_envs < 1:
        raise ValueError("--num-envs must be positive")
    output_dir = args.output_dir.resolve()
    audit_path = output_dir / "injection-audit.jsonl"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_sonic_release.py"),
        "metrics",
        "--runtime",
        args.runtime,
        "--num-envs",
        str(args.num_envs),
        "--output-dir",
        str(output_dir),
    ]
    for override in hydra_overrides(args.lateral_mps, audit_path):
        command.extend(("--override", override))
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path.unlink(missing_ok=True)
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
