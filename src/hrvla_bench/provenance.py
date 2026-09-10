"""Reproducibility metadata captured for each benchmark run."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import subprocess
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def collect_provenance(repo_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Collect immutable and host metadata without modifying the host."""
    return {
        "schema_version": "1.0",
        "suite_id": plan["suite_id"],
        "suite_sha256": plan["suite_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "repository_revision": git_revision(repo_root),
        "repository_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "controller_contract": plan["controller_contract"],
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
