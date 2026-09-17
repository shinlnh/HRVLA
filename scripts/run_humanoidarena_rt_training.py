#!/usr/bin/env python3
"""Train the frozen three-seed HumanoidArena ST-RT and STR-RT checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.humanoidarena_rt_training import (  # noqa: E402
    RT_METHODS,
    build_training_command,
    load_rt_lock,
    resource_blockers,
    run_directory,
    training_complete,
    validate_rt_inputs,
)
from hrvla_bench.humanoidarena_training import load_training_lock  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock", type=Path, default=ROOT / "config/humanoidarena-rt-training.lock.json"
    )
    parser.add_argument(
        "--common-lock",
        type=Path,
        default=ROOT / "config/humanoidarena-gr00t-training.lock.json",
    )
    parser.add_argument(
        "--common-selection",
        type=Path,
        default=ROOT / "_artifacts/retraining/humanoidarena-common/selection.json",
    )
    parser.add_argument("--python", type=Path, default=ROOT / "_vendor/Isaac-GR00T/.venv/bin/python")
    parser.add_argument("--method", action="append", choices=RT_METHODS)
    parser.add_argument("--seed", action="append", type=int, choices=(0, 1, 2))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    lock = load_rt_lock(args.lock.resolve())
    common_lock = load_training_lock(args.common_lock.resolve())
    common_selection = json.loads(args.common_selection.read_text(encoding="utf-8"))
    methods = args.method or list(RT_METHODS)
    common_step = validate_rt_inputs(
        ROOT, lock, common_lock, common_selection, tuple(methods)
    )
    seeds = args.seed or list(lock["training_seeds"])
    commands = [
        build_training_command(
            ROOT, args.python, lock, common_lock, common_step, method_id, seed
        )
        for method_id in methods
        for seed in seeds
    ]
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return 0
    blockers = resource_blockers(lock, Path(__file__).name)
    if blockers:
        for blocker in blockers:
            print(f"blocked: {blocker}", file=sys.stderr)
        return 3
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    for method_id in methods:
        for seed in seeds:
            directory = run_directory(ROOT, lock, method_id, seed)
            if training_complete(directory):
                print(f"skip complete {method_id}/seed-{seed}")
                continue
            if directory.exists():
                raise RuntimeError(f"refusing to overwrite partial RT training: {directory}")
            directory.mkdir(parents=True)
            command = build_training_command(
                ROOT, args.python, lock, common_lock, common_step, method_id, seed
            )
            with (directory / "train.log").open("x", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                    log.flush()
                returncode = process.wait()
            if returncode:
                return returncode
            if not training_complete(directory):
                print(f"completion marker missing: {method_id}/seed-{seed}", file=sys.stderr)
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
