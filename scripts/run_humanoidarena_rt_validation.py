#!/usr/bin/env python3
"""Evaluate 100/200/300 on validation and freeze one step per RT method."""

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
    build_validation_command,
    evaluation_complete,
    load_rt_lock,
    resource_blockers,
    run_directory,
    select_checkpoint,
    training_complete,
    validate_rt_inputs,
)
from hrvla_bench.humanoidarena_training import load_training_lock  # noqa: E402


def _write_once(path: Path, value: dict) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing == value:
            return
        raise FileExistsError(f"refusing to replace different frozen RT selection: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    lock = load_rt_lock(args.lock.resolve())
    common_lock = load_training_lock(args.common_lock.resolve())
    common_selection = json.loads(args.common_selection.read_text(encoding="utf-8"))
    methods = args.method or list(RT_METHODS)
    validate_rt_inputs(ROOT, lock, common_lock, common_selection, tuple(methods))
    for method_id in methods:
        for seed in lock["training_seeds"]:
            if not training_complete(run_directory(ROOT, lock, method_id, seed)):
                raise RuntimeError(f"RT training is incomplete: {method_id}/seed-{seed}")
    commands = [
        build_validation_command(ROOT, args.python, lock, method_id, seed, step)
        for method_id in methods
        for seed in lock["training_seeds"]
        for step in lock["training"]["candidate_steps"]
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
        {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    )
    for method_id in methods:
        expected = int(lock["methods"][method_id]["validation_episodes"])
        for seed in lock["training_seeds"]:
            for step in lock["training"]["candidate_steps"]:
                output = run_directory(ROOT, lock, method_id, seed) / f"validation/checkpoint-{step}"
                if evaluation_complete(output, expected):
                    continue
                if output.exists():
                    raise RuntimeError(f"refusing to overwrite partial RT validation: {output}")
                output.mkdir(parents=True)
                command = build_validation_command(
                    ROOT, args.python, lock, method_id, seed, step
                )
                with (output / "evaluation.log").open("x", encoding="utf-8") as log:
                    result = subprocess.run(
                        command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT
                    )
                if result.returncode:
                    return result.returncode
                if not evaluation_complete(output, expected):
                    return 2
        report = select_checkpoint(ROOT, lock, method_id)
        _write_once(
            ROOT / lock["output_root"] / lock["methods"][method_id]["output_name"] / "selection.json",
            report,
        )
        print(json.dumps({"method_id": method_id, "selected_step": report["selected_step"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
