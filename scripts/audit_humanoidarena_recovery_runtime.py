#!/usr/bin/env python3
"""Audit one live HumanoidArena recovery injector trace and its sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.plan import load_json  # noqa: E402
from hrvla_bench.recovery_runtime_audit import audit_recovery_runtime_trace  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        default=ROOT / "benchmark/suites/hrvla_recovery_v0.json",
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_recovery_runtime_trace(
        load_json(args.suite.resolve()),
        args.scenario,
        args.runtime_dir,
        args.result_json,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "scenario": report["scenario_id"],
                "audit_sha256": report["audit_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
