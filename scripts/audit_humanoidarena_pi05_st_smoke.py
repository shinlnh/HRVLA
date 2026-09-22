#!/usr/bin/env python3
"""Audit one PI0.5 ST live integration episode and save a visual frame."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.plan import canonical_sha256  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(output: Path) -> dict:
    episode_path = output / "episode.json"
    trace_path = output / "method/method-trace.jsonl"
    summary_path = output / "method/method-summary.json"
    server_log = output / "server.log"
    sim_log = output / "sim.log"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    decisions = [row for row in traces if row.get("event") == "instruction_selected"]
    if (summary != episode.get("hrvla_method")
            or summary.get("method_id") != "pi05_st"
            or summary.get("trace_sha256") != canonical_sha256(traces)
            or summary.get("policy_requests") != len(decisions)
            or not decisions or decisions[0].get("selected_skill_id") is None):
        raise ValueError("PI0.5 ST trace/summary/episode contract failed")
    if any(row.get("policy_action_dim") != 40 for row in decisions):
        raise ValueError("PI0.5 ST trace contains a non-action40 policy request")
    if episode.get("failure_reason") in {"interrupted", "process_error", "sim_error"}:
        raise ValueError("infrastructure-failed episode cannot pass integration audit")
    if episode.get("video_recorded") is not True:
        raise ValueError("PI0.5 ST smoke has no video evidence")
    first_prompt = decisions[0]["instruction"]
    if not any(
        "[lerobot_vla_server] first_infer" in line and f"task={first_prompt!r}" in line
        for line in server_log.read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        raise ValueError("HTTP server did not confirm the first ST instruction")
    video = Path(episode["video_path"]).resolve(strict=True)
    if not video.is_relative_to(output.resolve()) or video.stat().st_size < 1024:
        raise ValueError("episode video is missing or outside the audited output")
    frame = output / "first-second.png"
    if frame.exists():
        raise FileExistsError(f"refusing to overwrite existing frame: {frame}")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", "1", "-i", str(video),
         "-frames:v", "1", "-y", str(frame)],
        check=True, capture_output=True, text=True,
    )
    if not frame.is_file() or frame.stat().st_size < 1024:
        raise RuntimeError("ffmpeg did not decode a valid visual evidence frame")
    return {
        "schema_version": 1,
        "status": "integration_smoke_pass",
        "claim_boundary": "single bounded development episode; timeout is scored behavior, not a success or paired benchmark claim",
        "method_id": "pi05_st",
        "task_id": summary["task_id"],
        "episode_seed": episode["episode_seed"],
        "max_steps": episode["max_steps"],
        "episode_steps": episode["episode_steps"],
        "success": episode["success"],
        "failure_reason": episode["failure_reason"],
        "policy_requests": len(decisions),
        "selected_skills": [row.get("selected_skill_id") for row in decisions],
        "action_dim": 40,
        "implementation_revision": summary["implementation_revision"],
        "program_sha256": summary["program_sha256"],
        "episode_sha256": sha256(episode_path),
        "trace_sha256": sha256(trace_path),
        "summary_sha256": sha256(summary_path),
        "server_log_sha256": sha256(server_log),
        "sim_log_sha256": sha256(sim_log),
        "video_path": str(video),
        "video_sha256": sha256(video),
        "frame_path": str(frame),
        "frame_sha256": sha256(frame),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve(strict=True)
    audit_path = output / "audit.json"
    if audit_path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {audit_path}")
    result = audit(output)
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
