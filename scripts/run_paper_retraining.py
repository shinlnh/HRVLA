#!/usr/bin/env python3
"""Run the locked ST-RT and STR-RT training-seed matrix sequentially."""

from __future__ import annotations

import argparse
from pathlib import Path

from hrvla_bench.retraining import load_lock, run_matrix


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock", type=Path, default=ROOT / "config/paper-retraining.lock.json"
    )
    parser.add_argument(
        "--python", type=Path, default=ROOT / "_vendor/Isaac-GR00T/.venv/bin/python"
    )
    parser.add_argument("--method", choices=["ST-RT", "STR-RT", "all"], default="all")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    lock = load_lock(args.lock)
    methods = list(lock["methods"]) if args.method == "all" else [args.method]
    seeds = args.seed if args.seed is not None else list(lock["training_seeds"])
    return run_matrix(ROOT, args.python, lock, methods, seeds, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
