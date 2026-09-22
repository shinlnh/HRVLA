#!/usr/bin/env python3
"""Re-audit the seven bounded PI0.5-ST Isaac smokes at one code revision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TASK_IDS = {
    "pp_box": "pick_and_place_box",
    "boxing": "boxing",
    "doubledesk": "double_desk",
    "football": "football",
    "sit_sofa": "sit_sofa",
    "vision_navi": "visual_navigation",
    "open_door": "open_door",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"missing or hash-mismatched evidence: {path}")


def audit_matrix(root: Path, architecture_revision: str) -> dict:
    rows = []
    program_hashes = set()
    for task_key, task_id in TASK_IDS.items():
        folder = root / f"{task_key}-smoke-v3"
        audit = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
        episode = json.loads((folder / "episode.json").read_text(encoding="utf-8"))
        summary = json.loads((folder / "method/method-summary.json").read_text(encoding="utf-8"))
        trace = [
            json.loads(line)
            for line in (folder / "method/method-trace.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        decisions = [row for row in trace if row.get("event") == "instruction_selected"]
        if (
            audit.get("status") != "integration_smoke_pass"
            or audit.get("method_id") != "pi05_st"
            or audit.get("task_id") != task_id
            or audit.get("implementation_revision") != architecture_revision
            or audit.get("action_dim") != 40
            or audit.get("max_steps") != 60
            or audit.get("episode_steps") != 60
            or audit.get("policy_requests", 0) < 1
            or audit.get("policy_requests") != len(decisions)
            or episode.get("hrvla_method") != summary
            or any(row.get("policy_action_dim") != 40 for row in decisions)
            or episode.get("failure_reason") in {"interrupted", "process_error", "sim_error"}
        ):
            raise ValueError(f"invalid live integration record: {task_key}")
        for relative, field in (
            ("episode.json", "episode_sha256"),
            ("method/method-trace.jsonl", "trace_sha256"),
            ("method/method-summary.json", "summary_sha256"),
            ("server.log", "server_log_sha256"),
            ("sim.log", "sim_log_sha256"),
            ("first-second.png", "frame_sha256"),
        ):
            verify_file(folder / relative, audit[field])
        videos = list((folder / "videos").rglob(Path(audit["video_path"]).name))
        if len(videos) != 1:
            raise ValueError(f"expected exactly one matching video: {task_key}")
        verify_file(videos[0], audit["video_sha256"])
        first_prompt = decisions[0]["instruction"]
        if not any(
            "[lerobot_vla_server] first_infer" in line and f"task={first_prompt!r}" in line
            for line in (folder / "server.log").read_text(encoding="utf-8", errors="replace").splitlines()
        ):
            raise ValueError(f"server prompt disagrees with ST trace: {task_key}")
        program_hashes.add(audit["program_sha256"])
        rows.append({
            "task_key": task_key,
            "task_id": task_id,
            "episode_seed": audit["episode_seed"],
            "episode_steps": audit["episode_steps"],
            "policy_requests": audit["policy_requests"],
            "first_skill": decisions[0]["selected_skill_id"],
            "success": audit["success"],
            "failure_reason": audit["failure_reason"],
            "audit_sha256": sha256(folder / "audit.json"),
        })
    if len(program_hashes) != 1:
        raise ValueError("seven task routes used different ST programs")
    return {
        "schema_version": 1,
        "status": "integration_smoke_pass_7_of_7",
        "claim_boundary": "seven 60-step live integration smokes; not a paired, full-horizon benchmark",
        "architecture_revision": architecture_revision,
        "program_sha256": next(iter(program_hashes)),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite matrix audit: {args.output}")
    contract = json.loads((Path(__file__).resolve().parents[1] / "benchmark/v2_contract.json").read_text())
    result = audit_matrix(args.evidence_root, contract["architecture_revision"])
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
