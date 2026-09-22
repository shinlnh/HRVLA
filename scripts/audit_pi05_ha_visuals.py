#!/usr/bin/env python3
"""Bind checked-in PI0.5 HA sample videos to the raw 168-video manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/benchmark/external/pi05_cuda_int8_v1_amended_summary.json"
VIDEOS = ROOT / "results/benchmark/pi05-ha/videos"
TASKS = ("boxing", "doubledesk", "football", "pp_box", "sit_sofa", "vision_navi", "open_door")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit() -> dict:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    selected = summary["selected_video_manifest"]
    if len(selected) != 168:
        raise ValueError("source selected-video manifest is not complete")
    rows = []
    for task in TASKS:
        matches = [row for row in selected if f"/{task}/" in row["path"]]
        if len(matches) != 24:
            raise ValueError(f"expected 24 sampled source videos for {task}")
        source = matches[0]
        copy = VIDEOS / f"{task}.mp4"
        if not copy.is_file() or copy.stat().st_size < 1000 or sha256(copy) != source["sha256"]:
            raise ValueError(f"checked-in visual differs from frozen source: {task}")
        rows.append({
            "task": task,
            "selection_rule": "first lexicographic source video in the frozen manifest",
            "diagnostic_only": task == "open_door",
            "source_path": source["path"],
            "source_sha256": source["sha256"],
            "copy_path": str(copy.relative_to(ROOT)),
            "copy_sha256": sha256(copy),
            "copy_size_bytes": copy.stat().st_size,
        })
    if {path.stem for path in VIDEOS.glob("*.mp4")} != set(TASKS):
        raise ValueError("visual evidence directory contains an extra or missing task")
    return {
        "schema_version": 1,
        "audit_passed": True,
        "claim_boundary": "deterministic visual samples, not representative outcome statistics",
        "selected_video_count": len(selected),
        "checked_in_video_count": len(rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite visual audit: {args.output}")
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "videos": result["checked_in_video_count"]}))


if __name__ == "__main__":
    main()
