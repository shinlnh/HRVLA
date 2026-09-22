#!/usr/bin/env python3
"""Fail-closed audit of the frozen PI0.5 recovery oracle, including rejections.

This reports whether the 180-trial experiment executed correctly separately
from whether its pre-registered 20/20-per-scenario admission gate passed.
Behavioral failures are evidence and must not be discarded or rerolled.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORACLE_ROOT = ROOT / "_artifacts/HumanoidArena/recovery-admission/oracle-corrected-pi05-int8-v1"
DEFAULT_CAPTURE = ROOT / "_artifacts/HumanoidArena/recovery-admission/capture-manifest-corrected-pi05-int8-v1.json"
DEFAULT_LOCK = ROOT / "config/humanoidarena-recovery-oracle.lock.json"
DEFAULT_OUTPUT = ROOT / "results/benchmark/recovery/oracle_pi05_int8_v0_audit.json"
EXPECTED_ROUTES = {
    "pp_box": ("HOI_pp_box", "Move the box from the table onto the shelf."),
    "open_door": ("HSI_open_door", "Open the door."),
    "doubledesk": ("HOI_double_desk", "Put the hammer from the right table into the basket on the left table."),
    "football": ("HOI_football", "Kick the soccer ball into the goal."),
    "sit_sofa": ("HSI_sit_sofa", "Sit on the sofa."),
    "boxing": ("HSI_boxing", "Strike the green markers on the punching bag."),
    "vision_navi": ("HSI_vision_navi", "Avoid obstacles and move to the yellow marked area."),
}
INFRA_FAILURES = {"interrupted", "sim_error", "process_error"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trial_files(scenario_dir: Path, index: int) -> tuple[Path, Path]:
    records = list(scenario_dir.glob(f"attempt-*/trial-{index:04d}/oracle-record.json"))
    if len(records) != 1:
        raise ValueError(f"{scenario_dir.name} trial {index}: expected exactly one immutable record, got {len(records)}")
    episode = records[0].parent / "episode.json"
    if not episode.is_file():
        raise ValueError(f"{scenario_dir.name} trial {index}: episode JSON missing")
    return records[0], episode


def audit(oracle_root: Path, capture_path: Path, lock_path: Path) -> dict:
    capture, lock = _load(capture_path), _load(lock_path)
    progress_path = oracle_root / "progress.json"
    progress = _load(progress_path)
    if capture.get("status") != "runtime_and_snapshots_complete_oracle_admission_pending":
        raise ValueError("capture manifest is not complete")
    if progress.get("capture_manifest_sha256") != capture.get("manifest_sha256"):
        raise ValueError("oracle progress does not reference the locked capture")
    if progress.get("suite_sha256") != capture.get("suite_sha256") or lock.get("suite_sha256") != capture.get("suite_sha256"):
        raise ValueError("suite hashes differ")
    seeds = lock["rollout_seeds"]
    required = int(lock["protocol"]["required_trials"])
    threshold = int(lock["protocol"]["required_successes"])
    if len(seeds) != 9 or required != 20 or threshold != 20:
        raise ValueError("frozen nine-scenario, 20-of-20 oracle contract differs")
    if int(progress.get("scenarios_required", -1)) != len(seeds):
        raise ValueError("oracle progress scenario count differs")

    scenario_rows = []
    file_manifest = []
    global_reasons: Counter[str] = Counter()
    for scenario_id, scenario_seeds in seeds.items():
        if len(scenario_seeds) != required or len(set(scenario_seeds)) != required:
            raise ValueError(f"{scenario_id}: rollout seeds are missing or duplicated")
        scenario_dir = oracle_root / scenario_id
        report_path = scenario_dir / "oracle-report.json"
        report = _load(report_path)
        if report.get("scenario_id") != scenario_id or int(report.get("trials", -1)) != required:
            raise ValueError(f"{scenario_id}: scenario report is incomplete")
        counts: Counter[str] = Counter()
        for index, seed in enumerate(scenario_seeds):
            record_path, episode_path = _trial_files(scenario_dir, index)
            record, episode = _load(record_path), _load(episode_path)
            expected = {
                "scenario_id": scenario_id,
                "trial_index": index,
                "rollout_seed": seed,
                "capture_manifest_sha256": capture["manifest_sha256"],
                "suite_sha256": lock["suite_sha256"],
                "policy_backend": lock["resource_profile"]["policy_backend"],
                "policy_device": lock["resource_profile"]["policy_device"],
                "model_revision": lock["independence"]["model_revision"],
                "source_revision": lock["independence"]["source_revision"],
            }
            if any(record.get(key) != value for key, value in expected.items()):
                raise ValueError(f"{record_path}: frozen protocol field differs")
            if episode.get("episode_seed") != seed or _sha256(episode_path) != record.get("episode_result_sha256"):
                raise ValueError(f"{episode_path}: result seed or digest differs")
            if not record.get("start_state_restore_audit_sha256"):
                raise ValueError(f"{record_path}: restored start state was not audited")
            reason = str(record.get("failure_reason"))
            if reason in INFRA_FAILURES or reason not in {"success", "timeout", "fall", "failure_not_injected"}:
                raise ValueError(f"{record_path}: invalid or infrastructure outcome {reason}")
            if bool(record.get("success")) != (reason == "success"):
                raise ValueError(f"{record_path}: success flag/reason differs")
            protocol = record.get("protocol")
            if protocol == "online_failure":
                if reason != "failure_not_injected" and not record.get("runtime_audit_sha256"):
                    raise ValueError(f"{record_path}: online perturbation audit missing")
            elif protocol == "failure_start":
                if not record.get("failure_snapshot_sha256") or not record.get("failure_start_trial_audit_sha256"):
                    raise ValueError(f"{record_path}: failure-start snapshot audit missing")
            else:
                raise ValueError(f"{record_path}: unknown recovery protocol")
            if lock["protocol"].get("record_recovery_demonstrations") and not record.get("recovery_demonstration_arrays_sha256"):
                raise ValueError(f"{record_path}: recovery demonstration evidence missing")
            video_path = Path(str(record.get("video_path", ""))).resolve()
            if not record.get("video_recorded") or not video_path.is_file() or not video_path.is_relative_to(oracle_root.resolve()):
                raise ValueError(f"{record_path}: video evidence missing or outside oracle root")
            counts[reason] += 1
            global_reasons[reason] += 1
            file_manifest.extend([
                {"path": str(record_path.relative_to(ROOT)), "sha256": _sha256(record_path)},
                {"path": str(episode_path.relative_to(ROOT)), "sha256": _sha256(episode_path)},
                {"path": str(video_path.relative_to(ROOT)), "sha256": _sha256(video_path)},
            ])
        successes = counts["success"]
        admitted = successes == threshold
        if int(report.get("successes", -1)) != successes or bool(report.get("admitted")) != admitted:
            raise ValueError(f"{report_path}: report and raw trials disagree")
        scenario_rows.append({
            "scenario_id": scenario_id,
            "trials": required,
            "successes": successes,
            "reason_counts": dict(sorted(counts.items())),
            "admitted": admitted,
            "report_sha256": _sha256(report_path),
        })

    route_evidence = {}
    for task, expected in EXPECTED_ROUTES.items():
        log_path = oracle_root / "driver-logs" / task / "server.log"
        routes = re.findall(r"first_infer .*?task_name=([^ ]+) task='([^']+)'", log_path.read_text(encoding="utf-8"))
        if not routes or any(route != expected for route in routes):
            raise ValueError(f"{task}: policy language route is not invariant")
        route_evidence[task] = {"requests": len(routes), "task_name": expected[0], "instruction": expected[1]}

    admitted_count = sum(row["admitted"] for row in scenario_rows)
    if int(progress.get("scenarios_admitted", -1)) != admitted_count:
        raise ValueError("oracle progress disagrees with raw trial results")
    if sum(global_reasons.values()) != required * len(seeds) or len(file_manifest) != 3 * required * len(seeds):
        raise ValueError("oracle evidence count is incomplete")
    return {
        "schema_version": 1,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_complete": True,
        "admission_gate_passed": admitted_count == len(seeds),
        "status": "admitted" if admitted_count == len(seeds) else "completed_with_behavioral_rejections",
        "claim_boundary": "frozen independent PI0.5 recoverability witness; rejected trials are retained, not rerolled",
        "oracle_root": str(oracle_root.relative_to(ROOT)),
        "capture_manifest_path": str(capture_path.relative_to(ROOT)),
        "capture_manifest_sha256": _sha256(capture_path),
        "oracle_lock_sha256": _sha256(lock_path),
        "progress_sha256": _sha256(progress_path),
        "trials": required * len(seeds),
        "scenarios": len(seeds),
        "scenarios_admitted": admitted_count,
        "successes": global_reasons["success"],
        "reason_counts": dict(sorted(global_reasons.items())),
        "by_scenario": scenario_rows,
        "policy_routes": route_evidence,
        "evidence_manifest": sorted(file_manifest, key=lambda row: row["path"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--capture-manifest", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--oracle-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(args.oracle_root.resolve(), args.capture_manifest.resolve(), args.oracle_lock.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[recovery-oracle-audit] run_complete={report['run_complete']} admitted={report['scenarios_admitted']}/{report['scenarios']} output={args.output}")


if __name__ == "__main__":
    main()
