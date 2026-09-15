#!/usr/bin/env python3
"""Evaluate selected HA GR00T checkpoints on hidden data exactly once."""

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
    build_hidden_command,
    evaluation_complete,
    load_training_lock,
    resource_blockers,
    validate_selection,
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    lock = load_training_lock(args.lock)
    selection_path = ROOT / lock["output_root"] / "selection.json"
    if not selection_path.is_file():
        print(f"blocked: frozen validation selection is missing: {selection_path}", file=sys.stderr)
        return 3
    selected = validate_selection(json.loads(selection_path.read_text()), lock)
    commands = [
        build_hidden_command(ROOT, args.python, lock, seed, selected)
        for seed in lock["training_seeds"]
    ]
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return 0

    verify_training_inputs(ROOT, lock)
    blockers = resource_blockers(lock, Path(__file__).name)
    if blockers:
        for blocker in blockers:
            print(f"blocked: {blocker}", file=sys.stderr)
        return 3
    environment = os.environ.copy()
    environment.update(
        {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    )
    for seed, command in zip(lock["training_seeds"], commands):
        output = ROOT / lock["output_root"] / "hidden" / f"seed-{seed}"
        conditions = lock["hidden_evaluation"]["conditions"]
        if evaluation_complete(output, 140, conditions):
            print(f"skip complete hidden evaluation seed={seed}")
            continue
        if output.exists():
            raise RuntimeError(f"refusing to overwrite partial hidden evaluation: {output}")
        output.mkdir(parents=True)
        with (output / "evaluation.log").open("x", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if process.returncode:
            return process.returncode
        if not evaluation_complete(output, 140, conditions):
            print(f"incomplete hidden evaluation seed={seed}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
