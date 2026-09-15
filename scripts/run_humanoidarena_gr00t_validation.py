#!/usr/bin/env python3
"""Evaluate all locked checkpoints on validation and freeze one global step."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.humanoidarena_training import (  # noqa: E402
    build_validation_command,
    evaluation_complete,
    load_training_lock,
    resource_blockers,
    seed_directory,
    select_global_checkpoint,
    training_complete,
    verify_training_inputs,
)


def _run(command: list[str], log_path: Path, environment: dict[str, str]) -> int:
    with log_path.open("x", encoding="utf-8") as log:
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
        return process.wait()


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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    lock = load_training_lock(args.lock)
    commands = [
        build_validation_command(ROOT, args.python, lock, seed, step)
        for seed in lock["training_seeds"]
        for step in lock["training"]["candidate_steps"]
    ]
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return 0

    verify_training_inputs(ROOT, lock)
    for seed in lock["training_seeds"]:
        if not training_complete(seed_directory(ROOT, lock, seed)):
            raise RuntimeError(f"training is incomplete for seed={seed}")
    blockers = resource_blockers(lock, Path(__file__).name)
    if blockers:
        for blocker in blockers:
            print(f"blocked: {blocker}", file=sys.stderr)
        return 3

    environment = os.environ.copy()
    environment.update(
        {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    )
    evaluation_root = ROOT / lock["output_root"] / "validation"
    for seed in lock["training_seeds"]:
        for step in lock["training"]["candidate_steps"]:
            output = evaluation_root / f"seed-{seed}" / f"checkpoint-{step}"
            if evaluation_complete(output, 70, ["clean"]):
                print(f"skip complete validation seed={seed} step={step}")
                continue
            if output.exists():
                raise RuntimeError(f"refusing to overwrite partial validation: {output}")
            output.mkdir(parents=True)
            command = build_validation_command(ROOT, args.python, lock, seed, step)
            code = _run(command, output / "evaluation.log", environment)
            if code:
                return code
            if not evaluation_complete(output, 70, ["clean"]):
                print(f"incomplete validation seed={seed} step={step}", file=sys.stderr)
                return 2

    selection_path = ROOT / lock["output_root"] / "selection.json"
    if selection_path.exists():
        raise FileExistsError(f"refusing to replace frozen selection: {selection_path}")
    report = select_global_checkpoint(ROOT, lock, evaluation_root)
    selection_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_step": report["selected_step"],
                "selection_sha256": report["selection_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
