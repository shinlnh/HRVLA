#!/usr/bin/env python3
"""Freeze the common checkpoint step from the locked validation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.humanoidarena_training import (  # noqa: E402
    load_training_lock,
    select_global_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "config/humanoidarena-gr00t-training.lock.json",
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=ROOT / "_artifacts/retraining/humanoidarena-common/validation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "_artifacts/retraining/humanoidarena-common/selection.json",
    )
    args = parser.parse_args()
    report = select_global_checkpoint(ROOT, load_training_lock(args.lock), args.evaluation_root)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace frozen selection: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected_step": report["selected_step"], "selection_sha256": report["selection_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
