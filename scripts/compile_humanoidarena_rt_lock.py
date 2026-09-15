#!/usr/bin/env python3
"""Compile auditable candidate RT locks without editing the tracked lock in place."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.plan import load_json  # noqa: E402
from hrvla_bench.rt_lock_freeze import (  # noqa: E402
    freeze_adjudication,
    freeze_dataset_manifests,
    load_dataset_manifests,
)


def _write_once(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace RT lock candidate: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("adjudication", "datasets"))
    parser.add_argument("--lock", type=Path, default=ROOT / "config/humanoidarena-rt-training.lock.json")
    parser.add_argument("--report", type=Path, default=ROOT / "results/benchmark/retraining/subtask_video_adjudication.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    lock = load_json(args.lock.resolve())
    if args.phase == "adjudication":
        candidate = freeze_adjudication(lock, load_json(args.report.resolve()))
        default = ROOT / "_artifacts/HumanoidArena/benchmark-pipeline/rt-lock-after-adjudication.candidate.json"
    else:
        candidate = freeze_dataset_manifests(lock, load_dataset_manifests(ROOT, lock))
        default = ROOT / "_artifacts/HumanoidArena/benchmark-pipeline/rt-lock-ready-for-training.candidate.json"
    output = (args.output or default).resolve()
    _write_once(output, candidate)
    print(json.dumps({"output": str(output), "status": candidate["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
