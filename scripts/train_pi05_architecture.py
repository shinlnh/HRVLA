#!/usr/bin/env python3
"""Validate and launch one PI0.5 architecture fine-tuning run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess

from hrvla_bench.pi05_retraining import PI05TrainingJob, build_training_command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-id", choices=["pi05_st_rt", "pi05_str_rt"], required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--source-dataset", type=str,
                        help="e.g. HOI_pp_box; required for task-specific ST v3 export")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    job = PI05TrainingJob(
        method_id=args.method_id,
        dataset=args.dataset.resolve(),
        base_policy=args.base_policy.resolve(),
        output_dir=args.output_dir.resolve(),
        train_script=(args.lerobot_root / "src/lerobot/scripts/lerobot_train.py").resolve(),
        # Preserve the venv entry point: resolving this symlink bypasses its
        # site-packages and silently launches the system interpreter.
        python=args.python.absolute(),
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        source_dataset=args.source_dataset,
    )
    command = build_training_command(job)
    wrapper = Path(__file__).with_name("lerobot_pi05_low_mem_train.py").resolve()
    command[1:2] = [str(wrapper), str(job.train_script)]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    env = os.environ.copy()
    env["PYTHONPATH"] = str((args.lerobot_root / "src").resolve())
    env["HRVLA_PI05_TOKENIZER_DIR"] = str(args.tokenizer_dir.resolve(strict=True))
    return subprocess.call(command, env=env, cwd=args.lerobot_root)


if __name__ == "__main__":
    raise SystemExit(main())
