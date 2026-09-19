#!/usr/bin/env python3
"""Run the locked three-seed HA GR00T adaptation after the GPU becomes exclusive."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.humanoidarena_training import (  # noqa: E402
    build_training_command,
    load_training_lock,
    resource_blockers,
    seed_directory,
    training_complete,
    verify_training_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "config/humanoidarena-gr00t-training.lock.json",
    )
    parser.add_argument(
        "--python", type=Path, default=ROOT / "_vendor/Isaac-GR00T/.venv/bin/python"
    )
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    lock = load_training_lock(args.lock)
    seeds = args.seed if args.seed is not None else lock["training_seeds"]
    for seed in seeds:
        if seed not in lock["training_seeds"]:
            raise ValueError(f"seed {seed} is not locked")

    if args.dry_run:
        for seed in seeds:
            print(" ".join(build_training_command(ROOT, args.python, lock, seed)))
        return 0

    verify_training_inputs(ROOT, lock)
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
    for seed in seeds:
        directory = seed_directory(ROOT, lock, seed)
        if training_complete(directory):
            print(f"skip complete seed={seed}")
            continue
        if directory.exists():
            raise RuntimeError(
                f"refusing to overwrite partial seed directory: {directory}; audit it first"
            )
        directory.mkdir(parents=True)
        command = build_training_command(ROOT, args.python, lock, seed)
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
            return_code = process.wait()
        if return_code:
            return return_code
        if not training_complete(directory):
            print(f"completion marker missing for seed={seed}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
