#!/usr/bin/env python3
"""Resume the audited benchmark across its explicit post-external freeze gates.

This supervisor never edits or commits a tracked lock.  It executes every
deterministic stage that is currently authorized, writes the next candidate
lock, and waits for that candidate to be reviewed and committed on ``main``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = ROOT / "_artifacts/HumanoidArena/benchmark-pipeline"
DEFAULT_STATUS = PIPELINE_ROOT / "completion-status.json"
POST_STATUS = PIPELINE_ROOT / "status.json"
RT_LOCK = ROOT / "config/humanoidarena-rt-training.lock.json"
CHECKPOINT_LOCK = ROOT / "config/humanoidarena-internal-checkpoints.lock.json"
HIDDEN_GATE = ROOT / "config/humanoidarena-hidden-final.lock.json"
RELEASE_ROOT = ROOT / "_artifacts/HumanoidArena/release/humanoidarena-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _tracked_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _initial_status(path: Path) -> dict[str, Any]:
    if path.is_file():
        return _load(path)
    return {
        "schema_version": 1,
        "status": "waiting_for_post_external_pipeline",
        "source_revision_at_launch": _revision(),
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "stages": {},
    }


def _run_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = source_path + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[{_utc_now()}] command={json.dumps(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            raise
        return process.wait()


def _stage(
    state: dict[str, Any], status_path: Path, name: str, command: list[str]
) -> bool:
    if state["stages"].get(name, {}).get("status") == "complete":
        return True
    while not _tracked_clean():
        state["status"] = f"waiting_for_clean_worktree_before_{name}"
        state["updated_at_utc"] = _utc_now()
        _write(status_path, state)
        time.sleep(60)
    state["status"] = f"running_{name}"
    state["stages"][name] = {
        "status": "running",
        "command": command,
        "source_revision": _revision(),
        "started_at_utc": _utc_now(),
    }
    state["updated_at_utc"] = _utc_now()
    _write(status_path, state)
    code = _run_logged(command, status_path.parent / "completion-logs" / f"{name}.log")
    state["stages"][name].update(
        status="complete" if code == 0 else "failed",
        exit_code=code,
        finished_at_utc=_utc_now(),
    )
    state["status"] = f"{name}_{'complete' if code == 0 else 'failed'}"
    state["updated_at_utc"] = _utc_now()
    _write(status_path, state)
    return code == 0


def _wait_for(
    state: dict[str, Any],
    status_path: Path,
    label: str,
    predicate: Callable[[], bool],
    poll_seconds: int,
) -> None:
    while not predicate():
        state["status"] = label
        state["updated_at_utc"] = _utc_now()
        _write(status_path, state)
        time.sleep(poll_seconds)


def _post_external_ready() -> bool:
    if not POST_STATUS.is_file():
        return False
    status = _load(POST_STATUS)
    required = (
        "external_complete_audit",
        "common_gr00t_training",
        "common_gr00t_validation",
        "common_gr00t_hidden",
        "subtask_video_adjudication",
        "recovery_runtime_admission",
        "recovery_capture_manifest",
        "recovery_oracle",
        "admitted_suite",
        "recovery_training_dataset",
    )
    return (
        status.get("status") == "awaiting_audited_lock_freeze_and_rt_training"
        and all(status.get("stages", {}).get(name, {}).get("status") == "complete" for name in required)
    )


def _lock_status(path: Path, expected: str) -> bool:
    try:
        return _load(path).get("status") == expected
    except (OSError, json.JSONDecodeError):
        return False


def _adjudication_frozen() -> bool:
    try:
        policy = _load(RT_LOCK)["subtask_rt_dataset"]["temporal_video_adjudication"]
        report = _load(ROOT / policy["report_path"])
        return policy.get("status") == "ready" and policy.get("report_sha256") == report.get("audit_sha256")
    except (KeyError, OSError, json.JSONDecodeError):
        return False


def _probe_command(python: str) -> list[str]:
    plan = _load(RELEASE_ROOT / "release-plan.json")
    row = next(item for item in plan["artifacts"] if item["artifact_id"] == "common-seed-0")
    manifest = _load(ROOT / row["manifest_path"])
    return [
        python,
        "-u",
        "scripts/probe_humanoidarena_gr00t_coexistence.py",
        "--checkpoint",
        manifest["root"],
        "--manifest",
        row["manifest_path"],
    ]


def _audit_hidden_command(python: str) -> list[str]:
    root = "_artifacts/HumanoidArena/internal-benchmark/runs/hidden_final"
    return [
        python,
        "-m",
        "hrvla_bench.cli",
        "audit-evidence",
        "_artifacts/HumanoidArena/internal-benchmark/plans/hidden_final.plan.json",
        *[f"{root}/{method}/records.jsonl" for method in (
            "gr00t_sonic", "gr00t_st", "gr00t_st_rt", "gr00t_str", "gr00t_str_rt"
        )],
        "--output",
        "_artifacts/HumanoidArena/internal-benchmark/hidden-final-evidence-audit.json",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        parser.error("poll interval must be at least ten seconds")
    lock_path = args.status.resolve().with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another completion supervisor already owns the lock", file=sys.stderr)
        return 4
    lock_stream.write(str(os.getpid()) + "\n")
    lock_stream.flush()
    status_path = args.status.resolve()
    state = _initial_status(status_path)
    python = sys.executable
    groot_python = str(ROOT / "_vendor/Isaac-GR00T/.venv/bin/python")

    _wait_for(state, status_path, "waiting_for_post_external_pipeline", _post_external_ready, args.poll_seconds)
    if not _stage(
        state,
        status_path,
        "rt_adjudication_candidate",
        [python, "scripts/compile_humanoidarena_rt_lock.py", "adjudication"],
    ):
        return 2
    _wait_for(
        state,
        status_path,
        "waiting_for_reviewed_adjudication_lock_on_main",
        _adjudication_frozen,
        args.poll_seconds,
    )
    if not _stage(
        state,
        status_path,
        "subtask_training_dataset",
        [python, "scripts/prepare_humanoidarena_subtask_data.py"],
    ):
        return 2
    if not _stage(
        state,
        status_path,
        "rt_dataset_lock_candidate",
        [python, "scripts/compile_humanoidarena_rt_lock.py", "datasets"],
    ):
        return 2
    _wait_for(
        state,
        status_path,
        "waiting_for_reviewed_rt_dataset_lock_on_main",
        lambda: _lock_status(RT_LOCK, "ready_for_rt_training"),
        args.poll_seconds,
    )
    for name, command in (
        ("rt_training", [python, "-u", "scripts/run_humanoidarena_rt_training.py"]),
        ("rt_validation", [python, "-u", "scripts/run_humanoidarena_rt_validation.py"]),
        ("release_stage", [python, "scripts/release_humanoidarena_artifacts.py", "stage"]),
        ("release_publish", [groot_python, "-u", "scripts/release_humanoidarena_artifacts.py", "publish"]),
    ):
        if not _stage(state, status_path, name, command):
            return 2
    if not _stage(state, status_path, "coexistence_probe", _probe_command(groot_python)):
        return 2
    if not _stage(
        state,
        status_path,
        "checkpoint_lock_candidate",
        [python, "scripts/release_humanoidarena_artifacts.py", "compile-lock"],
    ):
        return 2
    _wait_for(
        state,
        status_path,
        "waiting_for_reviewed_checkpoint_lock_on_main",
        lambda: _lock_status(CHECKPOINT_LOCK, "ready_for_frozen_execution"),
        args.poll_seconds,
    )
    for name, command in (
        ("internal_plans", [python, "scripts/prepare_humanoidarena_internal_plans.py"]),
        ("internal_development", [python, "-u", "scripts/run_humanoidarena_internal_matrix.py", "--split", "development"]),
        ("internal_validation", [python, "-u", "scripts/run_humanoidarena_internal_matrix.py", "--split", "validation"]),
        ("hidden_final_gate_candidate", [python, "scripts/compile_humanoidarena_hidden_final_gate.py"]),
    ):
        if not _stage(state, status_path, name, command):
            return 2
    _wait_for(
        state,
        status_path,
        "waiting_for_reviewed_hidden_final_gate_on_main",
        lambda: _lock_status(HIDDEN_GATE, "ready_for_one_shot_hidden_final"),
        args.poll_seconds,
    )
    for name, command in (
        ("internal_hidden_final", [python, "-u", "scripts/run_humanoidarena_internal_matrix.py", "--split", "hidden_final"]),
        (
            "admitted_suite_claim_readiness",
            [python, "-m", "hrvla_bench.cli", "claim-readiness", "_artifacts/HumanoidArena/recovery-admission/hrvla_recovery_v0.admitted.json", "--output", "_artifacts/HumanoidArena/internal-benchmark/claim-readiness.json"],
        ),
        ("hidden_final_evidence_audit", _audit_hidden_command(python)),
        ("internal_final_report", [python, "scripts/render_humanoidarena_internal_results.py"]),
    ):
        if not _stage(state, status_path, name, command):
            return 2
    state["status"] = "awaiting_final_evidence_review_commit_and_frozen_tag"
    state["updated_at_utc"] = _utc_now()
    _write(status_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
