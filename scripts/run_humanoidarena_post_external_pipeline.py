#!/usr/bin/env python3
"""Durable, fail-closed handoff from the external matrix to fixed GPU stages.

The supervisor never edits Git-tracked locks or promotes a scientific claim.
It runs only stages whose inputs were frozen in advance, records every exit
code, and stops dependent stages after a failure while allowing independent
workstreams to continue.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATUS = ROOT / "_artifacts/HumanoidArena/benchmark-pipeline/status.json"
EXTERNAL_PROGRESS = (
    ROOT / "_artifacts/HumanoidArena/paper-baselines/pi05-sonic/progress.json"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_status(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "schema_version": 1,
        "status": "waiting_for_external_matrix",
        "source_revision_at_launch": revision,
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "external": {},
        "stages": {},
    }


def _tracked_worktree_is_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _external_complete() -> tuple[bool, dict[str, Any]]:
    if not EXTERNAL_PROGRESS.is_file():
        return False, {}
    progress = json.loads(EXTERNAL_PROGRESS.read_text(encoding="utf-8"))
    complete = (
        progress.get("complete") is True
        and int(progress.get("episodes_observed", -1))
        == int(progress.get("episodes_expected", -2))
        == 1680
    )
    return complete, progress


def _external_pids() -> list[int]:
    output = []
    own_pid = os.getpid()
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == own_pid:
            continue
        try:
            command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
            continue
        if "scripts/run_humanoidarena_baseline_matrix_fast.py" in command:
            output.append(int(process.name))
    return sorted(output)


def _run_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[{_utc_now()}] command={json.dumps(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
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


def _wait_for_external(status: dict[str, Any], status_path: Path, poll_seconds: int) -> bool:
    retries = int(status.get("external_resume_attempts", 0))
    while True:
        complete, progress = _external_complete()
        pids = _external_pids()
        status["external"] = progress
        status["updated_at_utc"] = _utc_now()
        _write_json_atomic(status_path, status)
        if complete and not pids:
            status["status"] = "external_matrix_complete"
            _write_json_atomic(status_path, status)
            return True
        if pids:
            status["status"] = "waiting_for_external_matrix"
            status["external_pids"] = pids
            _write_json_atomic(status_path, status)
            time.sleep(poll_seconds)
            continue
        if retries >= 3:
            status["status"] = "external_matrix_blocked_after_three_resume_attempts"
            status["external_pids"] = []
            _write_json_atomic(status_path, status)
            return False
        retries += 1
        status["external_resume_attempts"] = retries
        status["status"] = "restarting_incomplete_external_matrix"
        _write_json_atomic(status_path, status)
        code = _run_logged(
            [
                sys.executable,
                "-u",
                "scripts/run_humanoidarena_baseline_matrix_fast.py",
                "--cpu-threads",
                "24",
                "--compile-threads",
                "32",
                "--interop-threads",
                "2",
            ],
            status_path.parent / "external-resume.log",
        )
        status.setdefault("external_resume_exit_codes", []).append(code)
        _write_json_atomic(status_path, status)


def _stage(
    status: dict[str, Any],
    status_path: Path,
    name: str,
    command: list[str],
    *,
    dependencies: tuple[str, ...] = (),
) -> bool:
    existing = status["stages"].get(name)
    if existing and existing.get("status") == "complete":
        return True
    failed_dependencies = [
        dependency
        for dependency in dependencies
        if status["stages"].get(dependency, {}).get("status") != "complete"
    ]
    if failed_dependencies:
        status["stages"][name] = {
            "status": "blocked_by_dependency",
            "dependencies": failed_dependencies,
            "updated_at_utc": _utc_now(),
        }
        _write_json_atomic(status_path, status)
        return False
    if not _tracked_worktree_is_clean():
        status["stages"][name] = {
            "status": "blocked_dirty_tracked_worktree",
            "updated_at_utc": _utc_now(),
        }
        _write_json_atomic(status_path, status)
        return False
    status["status"] = f"running_{name}"
    status["stages"][name] = {
        "status": "running",
        "command": command,
        "started_at_utc": _utc_now(),
    }
    status["updated_at_utc"] = _utc_now()
    _write_json_atomic(status_path, status)
    code = _run_logged(command, status_path.parent / "logs" / f"{name}.log")
    status["stages"][name].update(
        status="complete" if code == 0 else "failed",
        exit_code=code,
        finished_at_utc=_utc_now(),
    )
    status["updated_at_utc"] = _utc_now()
    _write_json_atomic(status_path, status)
    return code == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        parser.error("--poll-seconds must be at least 10")
    status_path = args.status.resolve()
    status = _load_status(status_path)
    if not _tracked_worktree_is_clean():
        raise RuntimeError("post-external supervisor requires a clean tracked worktree")
    external_ok = _wait_for_external(status, status_path, args.poll_seconds)
    python = sys.executable
    groot_python = str(ROOT / "_vendor/Isaac-GR00T/.venv/bin/python")

    if external_ok:
        _stage(
            status,
            status_path,
            "external_complete_audit",
            [
                python,
                "scripts/summarize_humanoidarena_baseline_matrix.py",
                "--output",
                str(status_path.parent / "external-complete-summary.json"),
            ],
        )
    _stage(
        status,
        status_path,
        "common_gr00t_training",
        [python, "-u", "scripts/run_humanoidarena_gr00t_adaptation.py"],
    )
    _stage(
        status,
        status_path,
        "common_gr00t_validation",
        [python, "-u", "scripts/run_humanoidarena_gr00t_validation.py"],
        dependencies=("common_gr00t_training",),
    )
    _stage(
        status,
        status_path,
        "common_gr00t_hidden",
        [python, "-u", "scripts/run_humanoidarena_gr00t_hidden_eval.py"],
        dependencies=("common_gr00t_validation",),
    )
    _stage(
        status,
        status_path,
        "subtask_video_adjudication",
        [groot_python, "-u", "scripts/run_humanoidarena_subtask_video_adjudication.py"],
    )
    _stage(
        status,
        status_path,
        "recovery_runtime_admission",
        [python, "-u", "scripts/run_humanoidarena_recovery_admission.py"],
    )
    _stage(
        status,
        status_path,
        "recovery_capture_manifest",
        [python, "scripts/compile_humanoidarena_recovery_admission.py"],
        dependencies=("recovery_runtime_admission",),
    )
    _stage(
        status,
        status_path,
        "recovery_oracle",
        [python, "-u", "scripts/run_humanoidarena_recovery_oracle.py"],
        dependencies=("recovery_capture_manifest",),
    )
    _stage(
        status,
        status_path,
        "admitted_suite",
        [python, "scripts/compile_humanoidarena_admitted_suite.py"],
        dependencies=("recovery_oracle",),
    )
    _stage(
        status,
        status_path,
        "recovery_training_dataset",
        [
            python,
            "scripts/prepare_humanoidarena_recovery_data.py",
            "--suite",
            str(
                ROOT
                / "_artifacts/HumanoidArena/recovery-admission/"
                "hrvla_recovery_v0.admitted.json"
            ),
        ],
        dependencies=("admitted_suite",),
    )
    failures = [
        name
        for name, row in status["stages"].items()
        if row["status"] not in {"complete"}
    ]
    status["status"] = (
        "awaiting_audited_lock_freeze_and_rt_training"
        if not failures and external_ok
        else "post_external_pipeline_requires_audit"
    )
    status["incomplete_or_failed_stages"] = failures
    status["updated_at_utc"] = _utc_now()
    _write_json_atomic(status_path, status)
    return 0 if not failures and external_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
