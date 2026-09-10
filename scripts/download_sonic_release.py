#!/usr/bin/env python3
"""Download and verify only the artifacts required by the locked SONIC demo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "config" / "sonic-release.lock.json"
DEFAULT_DESTINATION = REPO_ROOT / "_vendor" / "GR00T-WholeBodyControl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    return parser.parse_args()


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required; install it in the SONIC runtime"
        ) from exc

    args = parse_args()
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    artifacts = lock["artifacts"]
    hub = lock["huggingface"]
    args.destination.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=str(hub["repo_id"]),
        repo_type=str(hub["repo_type"]),
        revision=str(hub["revision"]),
        allow_patterns=list(artifacts),
        local_dir=args.destination,
    )

    mismatches = []
    for relative, expected_hash in artifacts.items():
        artifact = args.destination / relative
        if not artifact.is_file():
            mismatches.append(f"missing: {relative}")
        else:
            actual_hash = sha256(artifact)
            if actual_hash != expected_hash:
                mismatches.append(
                    f"hash mismatch: {relative} ({actual_hash}, expected {expected_hash})"
                )
    if mismatches:
        raise RuntimeError("artifact validation failed:\n  " + "\n  ".join(mismatches))

    print(f"Verified {len(artifacts)} locked SONIC artifacts in {args.destination}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (json.JSONDecodeError, KeyError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
