#!/usr/bin/env python3
"""Fail closed if the checked-in PI0.5-ST HumanoidArena release drifts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "results/benchmark/pi05-st-ha"


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
        or manifest.get("successes") != 346
        or manifest.get("diagnostic_episodes") != 240
        or manifest.get("diagnostic_successes") != 2
    ):
        raise ValueError("release manifest has an unexpected denominator or result")
    for relative, expected in manifest["artifacts"].items():
        path = (package / relative).resolve()
        if not path.is_relative_to(package.resolve()):
            raise ValueError(f"release artifact escapes package: {relative}")
        require_hash(path, expected)

    primary = json.loads((package / "primary-audit.json").read_text(encoding="utf-8"))
    diagnostic = json.loads((package / "diagnostic-audit.json").read_text(encoding="utf-8"))
    comparison = json.loads((package / "comparison.json").read_text(encoding="utf-8"))
    overall = comparison["overall"]
    if (
        primary.get("audit_passed") is not True
        or primary.get("episodes_observed") != 1440
        or primary.get("successes") != 346
        or diagnostic.get("audit_passed") is not True
        or diagnostic.get("episodes_observed") != 240
        or diagnostic.get("successes") != 2
        or comparison.get("audit_passed") is not True
        or comparison.get("episodes_observed") != 1440
        or comparison.get("st_primary_audit_sha256") != manifest["artifacts"]["primary-audit.json"]
        or overall.get("baseline_successes") != 638
        or overall.get("st_successes") != 346
        or overall.get("st_minus_baseline_success_rate") != manifest["st_minus_baseline_success_rate"]
    ):
        raise ValueError("checked-in audits and paired comparison disagree")

    primary_samples = {row["task"]: row for row in comparison["sample_video_manifest"]}
    expected_primary = {"boxing", "doubledesk", "football", "pp_box", "sit_sofa", "vision_navi"}
    if set(primary_samples) != expected_primary:
        raise ValueError("paired comparison has an unexpected primary video set")
    for task, row in primary_samples.items():
        if sha256(package / "videos" / f"{task}.mp4") != row["sha256"]:
            raise ValueError(f"checked-in primary video differs from audited sample: {task}")
    diagnostic_samples = [row for row in diagnostic["episode_manifest"] if row.get("video_sha256")]
    selected = min(diagnostic_samples, key=lambda row: (row["mode"], row["seed"], row["repeat_idx"]))
    if sha256(package / "videos/open_door.mp4") != selected["video_sha256"]:
        raise ValueError("checked-in OpenDoor video differs from audited sample")
    if {path.stem for path in (package / "videos").glob("*.mp4")} != expected_primary | {"open_door"}:
        raise ValueError("release video directory contains an extra or missing task")
    return {
        "schema_version": 1,
        "audit_passed": True,
        "episodes_observed": 1440,
        "diagnostic_episodes": 240,
        "checked_in_videos": 7,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
