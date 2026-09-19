#!/usr/bin/env python3
"""Materialize frozen dev/validation/final plans from an admitted HA suite."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.internal_protocol import (  # noqa: E402
    build_internal_plans,
    protocol_summary,
)
from hrvla_bench.plan import load_json  # noqa: E402


def _write_once(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace frozen plan artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        default=ROOT
        / "_artifacts/HumanoidArena/recovery-admission/hrvla_recovery_v0.admitted.json",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "config/humanoidarena-internal-protocol.lock.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "_artifacts/HumanoidArena/internal-benchmark/plans",
    )
    args = parser.parse_args()

    lock = load_json(args.lock.resolve())
    plans = build_internal_plans(load_json(args.suite.resolve()), lock)
    output_root = args.output_root.resolve()
    for split_name, plan in plans.items():
        _write_once(output_root / f"{split_name}.plan.json", plan)
    summary = protocol_summary(plans, lock)
    _write_once(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
