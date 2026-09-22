#!/usr/bin/env python3
"""Run the OpenDoor diagnostic, strict audits, and paired comparison.

This unattended continuation never marks a branch complete or pushes a claim.
It records its state and leaves both raw and derived artifacts for review.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _running_primary(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except FileNotFoundError:
        return False
    return b"run_humanoidarena_pi05_st_matrix.py" in command


def _run_logged(command: list[str], log_path: Path, *, cwd: Path) -> None:
    with log_path.open("a", encoding="utf-8", buffering=1) as stream:
        stream.write("\nCOMMAND " + json.dumps(command) + "\n")
        result = subprocess.run(
            command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False,
        )
        stream.write(f"EXIT {result.returncode}\n")
    if result.returncode != 0:
        raise RuntimeError(f"command failed with {result.returncode}; see {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-pid", type=int, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--architecture-root", type=Path, required=True)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    args = parser.parse_args()
    runtime_root = args.runtime_root.resolve(strict=True)
    architecture_root = args.architecture_root.resolve(strict=True)
    primary_root = args.primary_root.resolve(strict=True)
    diagnostic_root = args.diagnostic_root.resolve()
    release_root = args.release_root.resolve()
    release_root.mkdir(parents=True, exist_ok=True)
    status_path = release_root / "continuation-status.json"

    def status(stage: str, **extra: object) -> None:
        _write_json(status_path, {
            "stage": stage,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "primary_pid_at_start": args.primary_pid,
            "primary_root": str(primary_root),
            "diagnostic_root": str(diagnostic_root),
            "release_root": str(release_root),
            **extra,
        })

    try:
        status("waiting_for_primary")
        while _running_primary(args.primary_pid):
            time.sleep(60)
        primary = _read_json(primary_root / "progress.json")
        if not primary.get("complete") or (
            primary.get("episodes_observed") != 1440
            or primary.get("cells_completed") != 72
        ):
            raise RuntimeError("primary matrix stopped before its 1,440 paired episodes")
        audit_script = ROOT / "scripts/audit_pi05_st_ha_matrix.py"
        primary_audit = release_root / "primary-audit.json"
        status("auditing_primary", primary_progress=primary)
        if primary_audit.exists():
            if not _read_json(primary_audit).get("audit_passed"):
                raise RuntimeError("existing primary audit is not valid")
        else:
            _run_logged([
                sys.executable, str(audit_script),
                "--runtime-root", str(runtime_root),
                "--matrix-root", str(primary_root),
                "--output", str(primary_audit),
            ], release_root / "primary-audit.log", cwd=ROOT)
        status("running_opendoor_diagnostic", primary_progress=primary)
        diagnostic_root.mkdir(parents=True, exist_ok=True)
        _run_logged([
            sys.executable, "-u",
            str(architecture_root / "scripts/run_humanoidarena_pi05_st_matrix.py"),
            "--runtime-root", str(runtime_root),
            "--output-root", str(diagnostic_root),
            "--tasks", "open_door", "--modes", "base_test", "semantic", "vision", "execution",
            "--seeds", "0", "1", "2", "--repeats", "20",
            "--policy-backend", "cuda_int8_weight_only",
            "--record-video-every-n", "10", "--step-log-every-n", "250",
        ], diagnostic_root / "driver.log", cwd=architecture_root)
        diagnostic = _read_json(diagnostic_root / "progress.json")
        if not diagnostic.get("complete") or (
            diagnostic.get("episodes_observed") != 240
            or diagnostic.get("cells_completed") != 12
        ):
            raise RuntimeError("OpenDoor diagnostic ended without all 240 episodes")
        status("auditing_diagnostic", primary_progress=primary, diagnostic_progress=diagnostic)
        diagnostic_audit = release_root / "diagnostic-audit.json"
        if diagnostic_audit.exists():
            if not _read_json(diagnostic_audit).get("audit_passed"):
                raise RuntimeError("existing diagnostic audit is not valid")
        else:
            _run_logged([
                sys.executable, str(audit_script),
                "--runtime-root", str(runtime_root),
                "--matrix-root", str(diagnostic_root),
                "--output", str(diagnostic_audit), "--diagnostic",
            ], release_root / "diagnostic-audit.log", cwd=ROOT)
        comparison_root = release_root / "comparison"
        comparison_path = comparison_root / "comparison.json"
        if comparison_path.exists():
            if (
                not _read_json(comparison_path).get("audit_passed")
                or not (comparison_root / "comparison.svg").is_file()
            ):
                raise RuntimeError("existing paired comparison is incomplete")
        else:
            status("compiling_paired_comparison")
            _run_logged([
                sys.executable, str(ROOT / "scripts/compile_pi05_st_ha_comparison.py"),
                "--runtime-root", str(runtime_root),
                "--primary-audit", str(primary_audit),
                "--output-root", str(comparison_root),
            ], release_root / "comparison.log", cwd=ROOT)
        status(
            "audits_and_comparison_passed_review_required",
            primary_audit=str(release_root / "primary-audit.json"),
            diagnostic_audit=str(release_root / "diagnostic-audit.json"),
            comparison=str(comparison_root / "comparison.json"),
            chart=str(comparison_root / "comparison.svg"),
        )
        return 0
    except Exception as exc:
        status("stopped_requires_review", error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
