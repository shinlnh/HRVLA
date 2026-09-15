#!/usr/bin/env python3
"""Compile an admitted HA suite from captures, predicates, and all oracle reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.plan import canonical_sha256, load_json  # noqa: E402
from hrvla_bench.recovery_release import build_admitted_suite  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--draft-suite",
        type=Path,
        default=ROOT / "benchmark/suites/hrvla_recovery_v0.json",
    )
    parser.add_argument(
        "--capture-manifest",
        type=Path,
        default=ROOT / "_artifacts/HumanoidArena/recovery-admission/capture-manifest.json",
    )
    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=ROOT / "_artifacts/HumanoidArena/recovery-admission/oracle",
    )
    parser.add_argument(
        "--predicate-audit",
        type=Path,
        default=ROOT / "results/humanoidarena/admission/predicate-source-audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "_artifacts/HumanoidArena/recovery-admission/hrvla_recovery_v0.admitted.json",
    )
    args = parser.parse_args()

    draft = load_json(args.draft_suite.resolve())
    scenario_ids = [
        scenario["id"] for task in draft["tasks"] for scenario in task["scenarios"]
    ]
    oracle_reports = {
        scenario_id: load_json(
            args.oracle_root.resolve() / scenario_id / "oracle-report.json"
        )
        for scenario_id in scenario_ids
    }
    released = build_admitted_suite(
        draft,
        load_json(args.capture_manifest.resolve()),
        oracle_reports,
        load_json(args.predicate_audit.resolve()),
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace an admitted suite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(released, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "status": released["status"],
                "draft_suite_sha256": released["admission_provenance"][
                    "draft_suite_sha256"
                ],
                "admitted_suite_sha256": canonical_sha256(released),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
