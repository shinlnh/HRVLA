#!/usr/bin/env python3
"""Re-audit recovery captures and compile their immutable snapshot manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.plan import load_json  # noqa: E402
from hrvla_bench.recovery_admission import (  # noqa: E402
    build_capture_manifest,
    discover_valid_captures,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        default=ROOT / "benchmark/suites/hrvla_recovery_v0.json",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=ROOT / "_artifacts/HumanoidArena/recovery-admission/runtime",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "_artifacts/HumanoidArena/recovery-admission/capture-manifest.json",
    )
    args = parser.parse_args()
    suite = load_json(args.suite.resolve())
    runtime_root = args.runtime_root.resolve()
    captures = discover_valid_captures(suite, runtime_root)
    manifest = build_capture_manifest(suite, captures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest_sha256": manifest["manifest_sha256"],
                "runtime": manifest["gates"]["runtime_injectors"],
                "initial_snapshots": manifest["gates"]["initial_snapshots"],
                "failure_snapshots": manifest["gates"]["failure_snapshots"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
