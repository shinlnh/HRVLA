#!/usr/bin/env python3
"""Create or verify the lock for final benchmark checkpoint artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hrvla_bench.artifacts import build_artifact_lock, verify_artifact_lock  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify"))
    parser.add_argument(
        "--lock",
        type=Path,
        default=REPO_ROOT / "config" / "benchmark-artifacts.lock.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "create":
        lock = build_artifact_lock(REPO_ROOT)
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        args.lock.write_text(
            json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"artifact lock: {args.lock}")
        print(f"groups: {len(lock['artifacts'])}")
        return 0

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    errors = verify_artifact_lock(lock, REPO_ROOT)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"artifact lock: valid ({len(lock['artifacts'])} groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
