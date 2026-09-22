#!/usr/bin/env python3
"""Route a single PI0.5+SONIC episode through the frozen ST language program.

This is an architecture-integration runner, not a benchmark matrix. It uses
the pinned upstream Isaac evaluator and an already running PI0.5 HTTP server.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.humanoidarena_method_hooks import install_method_hooks  # noqa: E402
from hrvla_bench.humanoidarena_method_runtime import load_method_programs  # noqa: E402


HUMANOIDARENA_REVISION = "68479287a784a69be9ce6ad739311d2f11f75ef9"
ISAACLAB_REVISION = "46dff135f44683f031edf346e544fcfd8456b2bb"
TASK_IDS = {
    "Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody": "boxing",
    "Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody": "double_desk",
    "Isaac-Move-Football-Single-G129-Dex3-Wholebody": "football",
    "Isaac-Move-Open-Door-G129-Dex3-Wholebody": "open_door",
    "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby": "pick_and_place_box",
    "Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody": "sit_sofa",
    "Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody": "visual_navigation",
}


def _git_revision(path: Path, *, require_clean: bool) -> str:
    revision = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if require_clean:
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--short", "--untracked-files=no"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if dirty:
            raise RuntimeError("PI0.5 ST evidence requires a clean implementation worktree")
    return revision


def validate_invocation(remaining: list[str], programs: dict) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episode_batch_json", default="")
    parser.add_argument("--lerobot_server_url", required=True)
    parser.add_argument("--enable_cameras", action="store_true")
    values, _ = parser.parse_known_args(remaining)
    if values.episode_batch_json:
        raise ValueError("PI0.5 ST integration runner accepts one episode, not a batch")
    if values.task not in TASK_IDS:
        raise ValueError(f"task is outside the seven-task PI0.5 ST contract: {values.task}")
    if {row["task_id"] for row in programs["tasks"]} != set(TASK_IDS.values()):
        raise ValueError("PI0.5 ST program tasks differ from the evaluator task map")
    if not values.lerobot_server_url.startswith("http://127.0.0.1:"):
        raise ValueError("PI0.5 ST integration requires a loopback HTTP policy server")
    if not values.enable_cameras:
        raise ValueError("PI0.5 ST integration requires the front camera")
    return TASK_IDS[values.task]


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--method-output-dir", type=Path, required=True)
    parser.add_argument(
        "--method-programs", type=Path,
        default=ROOT / "benchmark/humanoidarena_method_programs.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    args, remaining = parser.parse_known_args()
    programs = load_method_programs(args.method_programs.resolve(strict=True))
    task_id = validate_invocation(remaining, programs)
    runtime_root = args.runtime_root.resolve(strict=True)
    humanoidarena = runtime_root / "_vendor/HumanoidArena"
    isaaclab = runtime_root / "_vendor/IsaacLab-v2.2.0"
    evaluator = (
        humanoidarena
        / "isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/sim_eval_vla.py"
    )
    if not evaluator.is_file():
        raise FileNotFoundError(f"pinned HumanoidArena evaluator is missing: {evaluator}")
    if _git_revision(humanoidarena, require_clean=False) != HUMANOIDARENA_REVISION:
        raise RuntimeError("HumanoidArena revision differs from the frozen PI0.5 baseline")
    if _git_revision(isaaclab, require_clean=False) != ISAACLAB_REVISION:
        raise RuntimeError("Isaac Lab revision differs from the frozen PI0.5 baseline")
    implementation_revision = _git_revision(ROOT, require_clean=True)
    output = args.method_output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"method output is not empty: {output}")
    print(json.dumps({
        "method_id": "pi05_st", "task_id": task_id,
        "implementation_revision": implementation_revision,
        "evaluator": str(evaluator), "method_output_dir": str(output),
    }, sort_keys=True), flush=True)
    if args.dry_run:
        return 0
    spec = importlib.util.spec_from_file_location("hrvla_pi05_st_upstream_evaluator", evaluator)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pinned evaluator: {evaluator}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    install_method_hooks(
        module, programs, method_id="pi05_st", task_id=task_id,
        output_dir=output, implementation_revision=implementation_revision,
    )
    sys.argv = [sys.argv[0], *remaining]
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
