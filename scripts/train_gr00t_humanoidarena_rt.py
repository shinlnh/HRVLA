#!/usr/bin/env python3
"""Train a GR00T RT method only after its 40-D data contract is validated."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_subtask.humanoidarena_retraining import GR00TRetrainJob, build_command  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-id", choices=["gr00t_st_rt", "gr00t_str_rt"], required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    job = GR00TRetrainJob(
        method_id=args.method_id,
        python=args.python.resolve(),
        train_script=ROOT / "scripts/train_gr00t_vla_adapter.py",
        dataset=args.dataset.resolve(),
        base_checkpoint=args.base_checkpoint.resolve(),
        modality_config=ROOT / "config/g1_humanoidarena_refpose_config.py",
        output_dir=args.output_dir.resolve(),
        seed=args.seed,
        steps=args.steps,
        workers=args.workers,
        batch_size=args.batch_size,
        accumulation=args.accumulation,
    )
    command = build_command(job)
    print(shlex.join(command), flush=True)
    return 0 if args.dry_run else subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
