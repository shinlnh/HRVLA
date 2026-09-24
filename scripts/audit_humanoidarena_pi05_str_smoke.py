#!/usr/bin/env python3
"""Fail-closed audit for one live PI0.5-STR recovery smoke episode."""

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
from hrvla_bench.recovery_runtime_audit import audit_recovery_runtime_trace  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(output: Path, suite_path: Path, scenario_id: str) -> dict:
    episode_path = output / "episode.json"
    method_dir = output / "method"
    trace_path = method_dir / "method-trace.jsonl"
    summary_path = method_dir / "method-summary.json"
    recovery_trace_path = method_dir / "runtime-trace.jsonl"
    recovery_summary_path = method_dir / "runtime-summary.json"
    server_log = output / "server.log"
    sim_log = output / "sim.log"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recovery = json.loads(recovery_summary_path.read_text(encoding="utf-8"))
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    traces = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    decisions = [row for row in traces if row.get("event") == "instruction_selected"]
    recovery_decisions = [row for row in decisions if row.get("recovery_active") is True]
    if (summary != episode.get("hrvla_method")
            or recovery != episode.get("hrvla_recovery")
            or summary.get("method_id") != "pi05_str"
            or summary.get("trace_sha256") != canonical_sha256(traces)
            or summary.get("policy_requests") != len(decisions)
            or summary.get("recovery_decisions") != len(recovery_decisions)
            or not recovery_decisions
            or recovery.get("scenario_id") != scenario_id
            or recovery.get("triggered") is not True):
        raise ValueError("PI0.5-STR method/recovery contract failed")
    if any(row.get("policy_action_dim") != 40 for row in decisions):
        raise ValueError("PI0.5-STR trace contains a non-action40 policy request")
    first_recovery_step = summary.get("first_recovery_decision_control_step")
    if not isinstance(first_recovery_step, int):
        raise ValueError("PI0.5-STR did not record its first recovery decision")
    if episode.get("failure_reason") in {"interrupted", "process_error", "sim_error"}:
        raise ValueError("infrastructure-failed episode cannot pass integration audit")
    if episode.get("video_recorded") is not True:
        raise ValueError("PI0.5-STR smoke has no video evidence")
    server_text = server_log.read_text(encoding="utf-8", errors="replace")
    first_prompt = decisions[0]["instruction"]
    if not any(
        "[lerobot_vla_server] first_infer" in line and f"task={first_prompt!r}" in line
        for line in server_text.splitlines()
    ):
        raise ValueError("HTTP server did not confirm the live method session")
    runtime_audit = audit_recovery_runtime_trace(
        suite, scenario_id, method_dir, episode_path
    )
    runtime_audit_path = output / "runtime-audit.json"
    runtime_audit_path.write_text(
        json.dumps(runtime_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    video = Path(episode["video_path"]).resolve(strict=True)
    if not video.is_relative_to(output.resolve()) or video.stat().st_size < 1024:
        raise ValueError("episode video is missing or outside the audited output")
    frame = output / "recovery-evidence.png"
    if frame.exists():
        raise FileExistsError(f"refusing to overwrite existing frame: {frame}")
    seek_s = max(0.0, first_recovery_step / 50.0)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{seek_s:.3f}",
         "-i", str(video), "-frames:v", "1", "-y", str(frame)],
        check=True, capture_output=True, text=True,
    )
    if not frame.is_file() or frame.stat().st_size < 1024:
        raise RuntimeError("ffmpeg did not decode a recovery evidence frame")
    return {
        "schema_version": 1,
        "status": "integration_smoke_pass",
        "claim_boundary": "single live injected-recovery episode; not a paired benchmark claim",
        "method_id": "pi05_str",
        "task_id": summary["task_id"],
        "scenario_id": scenario_id,
        "episode_seed": episode["episode_seed"],
        "success": episode["success"],
        "failure_reason": episode["failure_reason"],
        "policy_requests": len(decisions),
        "recovery_decisions": len(recovery_decisions),
        "first_recovery_decision_control_step": first_recovery_step,
        "action_dim": 40,
        "episode_sha256": sha256(episode_path),
        "method_trace_sha256": sha256(trace_path),
        "method_summary_sha256": sha256(summary_path),
        "recovery_trace_sha256": sha256(recovery_trace_path),
        "recovery_summary_sha256": sha256(recovery_summary_path),
        "runtime_audit_sha256": sha256(runtime_audit_path),
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
    parser.add_argument("--suite", type=Path, default=ROOT / "benchmark/suites/hrvla_recovery_v0.json")
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve(strict=True)
    audit_path = output / "audit.json"
    if audit_path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {audit_path}")
    result = audit(output, args.suite.resolve(strict=True), args.scenario)
    audit_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
