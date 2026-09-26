#!/usr/bin/env python3
"""Fail closed if the checked-in PI0.5-STR HumanoidArena release drifts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "results/benchmark/pi05-str-ha"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"missing or hash-mismatched release artifact: {path}")


def audit(package: Path = PACKAGE) -> dict:
    manifest = json.loads((package / "result_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("audit_passed") is not True
        or manifest.get("episodes_observed") != 1440
        or manifest.get("successes") != 381
        or manifest.get("baseline_successes") != 638
    ):
        raise ValueError("release manifest has an unexpected denominator or result")
    for relative, expected in manifest["artifacts"].items():
        path = (package / relative).resolve()
        if not path.is_relative_to(package.resolve()):
            raise ValueError(f"release artifact escapes package: {relative}")
        require_hash(path, expected)

    primary = json.loads((package / "primary-audit.json").read_text(encoding="utf-8"))
    comparison = json.loads((package / "comparison.json").read_text(encoding="utf-8"))
    overall = comparison["overall"]
    if (
        primary.get("audit_passed") is not True
        or primary.get("method_id") != "pi05_str"
        or primary.get("episodes_observed") != 1440
        or primary.get("successes") != 381
        or comparison.get("audit_passed") is not True
        or comparison.get("episodes_observed") != 1440
        or comparison.get("str_primary_audit_sha256") != manifest["artifacts"]["primary-audit.json"]
        or overall.get("baseline_successes") != 638
        or overall.get("str_successes") != 381
        or overall.get("str_minus_baseline_success_rate") != manifest["str_minus_baseline_success_rate"]
    ):
        raise ValueError("checked-in audit and paired comparison disagree")

    samples = {row["task"]: row for row in comparison["sample_video_manifest"]}
    expected_tasks = {"boxing", "doubledesk", "football", "pp_box", "sit_sofa", "vision_navi"}
    if set(samples) != expected_tasks:
        raise ValueError("paired comparison has an unexpected video set")
    for task, row in samples.items():
        if sha256(package / "videos" / f"{task}.mp4") != row["sha256"]:
            raise ValueError(f"checked-in video differs from audited sample: {task}")
    if {path.stem for path in (package / "videos").glob("*.mp4")} != expected_tasks:
        raise ValueError("release video directory contains an extra or missing task")
    return {
        "schema_version": 1,
        "audit_passed": True,
        "episodes_observed": 1440,
        "checked_in_videos": 6,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
