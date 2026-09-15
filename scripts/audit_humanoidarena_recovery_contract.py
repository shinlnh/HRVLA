#!/usr/bin/env python3
"""Audit exact upstream sources behind the draft recovery success predicates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.humanoidarena_recovery_contract import (  # noqa: E402
    audit_recovery_contract,
)
from hrvla_bench.plan import load_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, default=ROOT / "benchmark/suites/hrvla_recovery_v0.json"
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT / "_vendor/HumanoidArena/isaaclab_twist2_g1",
    )
    parser.add_argument(
        "--expected-revision",
        default="68479287a784a69be9ce6ad739311d2f11f75ef9",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/humanoidarena/admission/predicate-source-audit.json",
    )
    args = parser.parse_args()
    report = audit_recovery_contract(
        load_json(args.suite), args.source_root.resolve(), args.expected_revision
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "task_predicates": len(report["tasks"]),
                "draft_scenarios": len(report["scenarios"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
