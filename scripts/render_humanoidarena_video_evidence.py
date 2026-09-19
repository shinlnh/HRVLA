#!/usr/bin/env python3
"""Render hash-bound representative simulator contact sheets for the paper."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.humanoidarena_release import file_sha256  # noqa: E402
from hrvla_bench.plan import canonical_sha256  # noqa: E402


FRACTIONS = (0.2, 0.5, 0.8)


def _inside_root(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(ROOT.resolve())
    return resolved


def _relative(path: Path) -> str:
    return _inside_root(path).relative_to(ROOT.resolve()).as_posix()


def _video_from_record(record: dict[str, Any]) -> Path | None:
    value = record.get("video_path")
    if not isinstance(value, str) or not value:
        return None
    try:
        path = _inside_root(Path(value))
    except (OSError, ValueError):
        return None
    return path if path.is_file() else None


def choose_representatives(rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    """Choose one deterministic real video per requested comparison cell."""

    selected = {}
    for row in sorted(rows, key=lambda item: json.dumps(item, sort_keys=True, default=str)):
        video = _video_from_record(row)
        if video is None:
            continue
        key = tuple(str(row.get(field, "")) for field in key_fields)
        candidate = {**row, "_resolved_video": str(video)}
        current = selected.get(key)
        if current is None or (bool(candidate.get("success")), str(video)) > (
            bool(current.get("success")), current["_resolved_video"]
        ):
            selected[key] = candidate
    return [selected[key] for key in sorted(selected)]


def _external_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*/*/seed-*/episodes/*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        relative = path.relative_to(root)
        value.update(
            evidence_group="external",
            mode=relative.parts[0],
            task_key=relative.parts[1],
            record_path=_relative(path),
        )
        rows.append(value)
    return choose_representatives(rows, ("mode", "task_key"))


def _recovery_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*/attempt-*/trial-*/oracle-record.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            manifest = json.loads(
                (path.parent / "recovery-demonstration.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        value.update(
            evidence_group="recovery_oracle",
            scenario_id=str(value.get("scenario_id") or path.parts[-4]),
            video_path=manifest.get("video_path"),
            record_path=_relative(path),
        )
        rows.append(value)
    return choose_representatives(rows, ("scenario_id",))


def _internal_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*/records.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            value = json.loads(line)
            value.update(
                evidence_group="internal_hidden_final",
                record_path=_relative(path),
            )
            rows.append(value)
    return choose_representatives(rows, ("method_id", "protocol"))


def _duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = float(result.stdout.strip())
    if value <= 0:
        raise ValueError(f"video duration is non-positive: {path}")
    return value


def _frame(path: Path, timestamp: float) -> Image.Image:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{timestamp:.6f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return Image.open(BytesIO(result.stdout)).convert("RGB")


def _label(row: dict[str, Any]) -> str:
    group = row["evidence_group"]
    if group == "external":
        identity = f"{row['mode']} / {row['task_key']}"
    elif group == "recovery_oracle":
        identity = row["scenario_id"]
    else:
        identity = f"{row['method_id']} / {row['protocol']}"
    outcome = "success" if row.get("success") else str(row.get("failure_reason", "failure"))
    return f"{group}: {identity} — {outcome}"


def _filename(row: dict[str, Any]) -> str:
    label = _label(row).lower()
    return "".join(character if character.isalnum() else "-" for character in label).strip("-") + ".png"


def _render_one(payload: tuple[dict[str, Any], str]) -> dict[str, Any]:
    row, building_value = payload
    video = Path(row["_resolved_video"])
    building = Path(building_value)
    duration = _duration(video)
    timestamps = [duration * value for value in FRACTIONS]
    frames = [_frame(video, timestamp) for timestamp in timestamps]
    target_height = 360
    resized = []
    for frame in frames:
        width = round(frame.width * target_height / frame.height)
        resized.append(frame.resize((width, target_height), Image.Resampling.LANCZOS))
    padding = 12
    title_height = 56
    canvas = Image.new(
        "RGB",
        (
            sum(frame.width + 4 for frame in resized) + padding * (len(resized) + 1),
            target_height + title_height + padding,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, 10), _label(row), fill="black")
    x = padding
    for frame, timestamp in zip(resized, timestamps, strict=True):
        framed = ImageOps.expand(frame, border=2, fill="#2d3748")
        canvas.paste(framed, (x, title_height))
        draw.text((x + 6, title_height + 6), f"t={timestamp:.1f}s", fill="white", stroke_width=2, stroke_fill="black")
        x += framed.width + padding
    destination = building / _filename(row)
    canvas.save(destination, format="PNG", optimize=True)
    return {
        "evidence_group": row["evidence_group"],
        "identity": _label(row),
        "outcome": "success" if row.get("success") else str(row.get("failure_reason", "failure")),
        "record_path": row["record_path"],
        "video_path": _relative(video),
        "video_sha256": file_sha256(video),
        "duration_seconds": duration,
        "sample_fractions": list(FRACTIONS),
        "sample_timestamps_seconds": timestamps,
        "contact_sheet": destination.name,
        "contact_sheet_sha256": file_sha256(destination),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, default=ROOT / "_artifacts/HumanoidArena/paper-baselines/pi05-sonic")
    parser.add_argument("--recovery-root", type=Path, default=ROOT / "_artifacts/HumanoidArena/recovery-admission/oracle")
    parser.add_argument("--internal-root", type=Path, default=ROOT / "_artifacts/HumanoidArena/internal-benchmark/runs/hidden_final")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/humanoidarena/video-evidence")
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace video evidence: {output}")
    building = output.with_name("." + output.name + ".building")
    if building.exists():
        raise FileExistsError(f"stale video evidence build requires audit: {building}")
    building.mkdir(parents=True)
    rows = [
        *_external_rows(args.external_root.resolve()),
        *_recovery_rows(args.recovery_root.resolve()),
        *_internal_rows(args.internal_root.resolve()),
    ]
    counts = {group: sum(row["evidence_group"] == group for row in rows) for group in (
        "external", "recovery_oracle", "internal_hidden_final"
    )}
    if counts != {"external": 28, "recovery_oracle": 9, "internal_hidden_final": 15}:
        raise ValueError(f"representative video matrix is incomplete: {counts}")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(_render_one, ((row, str(building)) for row in rows)))
    core = {
        "schema_version": 1,
        "claim_boundary": "representative visual evidence; quantitative claims come from complete JSONL matrices",
        "selection": "one deterministic available video per external task/mode, recovery scenario, and internal method/protocol; prefer success",
        "counts": counts,
        "records": sorted(records, key=lambda row: (row["evidence_group"], row["identity"])),
    }
    manifest = {**core, "manifest_sha256": canonical_sha256(core)}
    (building / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(building, output)
    print(json.dumps({"contact_sheets": len(records), "manifest_sha256": manifest["manifest_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
