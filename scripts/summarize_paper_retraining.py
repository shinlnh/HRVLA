#!/usr/bin/env python3
"""Summarize the locked three-seed ST-RT and STR-RT evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path

from hrvla_bench.paper_results import summarize, write_summary


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=ROOT / "_artifacts/retraining/paper-seeds/evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/retraining/paper-seeds",
    )
    args = parser.parse_args()
    write_summary(summarize(args.evaluation_root, [0, 1, 2]), args.output_dir)


if __name__ == "__main__":
    main()
