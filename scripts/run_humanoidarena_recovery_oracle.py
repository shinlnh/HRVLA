#!/usr/bin/env python3
"""Run the frozen 20/20 independent recoverability-witness protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.plan import canonical_sha256, load_json  # noqa: E402
from hrvla_bench.recovery_oracle import (  # noqa: E402
    evaluate_oracle_trials,
    validate_oracle_lock,
)
from hrvla_bench.recovery_restore import (  # noqa: E402
    audit_failure_start_trial,
    validate_restore_audit,
)
from hrvla_bench.recovery_runtime_audit import audit_recovery_runtime_trace  # noqa: E402


FAST_RUNNER_PATH = ROOT / "scripts/run_humanoidarena_baseline_matrix_fast.py"
RECOVERY_RUNNER = ROOT / "scripts/run_humanoidarena_recovery_episode.py"


def _load_fast_runner():
    spec = importlib.util.spec_from_file_location("hrvla_fast_matrix_oracle", FAST_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import orchestration primitives: {FAST_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FAST = _load_fast_runner()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scenario_rows(suite: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task in suite["tasks"]:
        task_key = task["humanoidarena_task_key"]
        if task_key not in FAST.TASKS:
            raise ValueError(f"task key is absent from the locked runner: {task_key}")
        for scenario in task["scenarios"]:
            rows.append(
                {
                    "task_id": task["id"],
                    "task_key": task_key,
                    "scenario_id": scenario["id"],
                    "protocol": scenario["protocol"],
                }
            )
    return rows


def _capture_row(capture_manifest: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    return next(
        row for row in capture_manifest["scenarios"] if row["scenario_id"] == scenario_id
    )


def _start_snapshot(
    capture_manifest: dict[str, Any],
    row: dict[str, Any],
    runtime_root: Path,
) -> tuple[Path, str]:
    capture = _capture_row(capture_manifest, row["scenario_id"])
    prefix = "failure" if row["protocol"] == "failure_start" else "initial"
    relative = capture[f"{prefix}_snapshot_path"]
    expected_sha256 = capture[f"{prefix}_snapshot_sha256"]
    if not isinstance(relative, str) or not isinstance(expected_sha256, str):
        raise ValueError(f"{row['scenario_id']}: capture manifest lacks {prefix} snapshot")
    path = (runtime_root / relative).resolve()
    try:
        path.relative_to(runtime_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{row['scenario_id']}: snapshot path escapes runtime root") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, expected_sha256


def _episode_job(
    task_key: str,
    trial_dir: Path,
    model_path: Path,
    trial_index: int,
    rollout_seed: int,
) -> dict[str, Any]:
    spec = FAST.TASKS[task_key]
    return {
        "seed": 0,
        "repeat_idx": int(trial_index),
        "episode_seed": int(rollout_seed),
        "episode_index": int(trial_index),
        "result_json": str(trial_dir / "episode.json"),
        "success_video_dir": str(trial_dir / "videos" / "success"),
        "failure_video_dir": str(trial_dir / "videos" / "failure"),
        "recording_save_dir": str(trial_dir / "recordings"),
        "model_label": FAST._model_label(model_path),
        "eval_model_path": str(model_path),
        "max_steps": int(spec["max_steps"]),
        "video_fps": 30,
        "post_termination_record_steps": 10,
    }


def build_oracle_command(
    *,
    suite_path: Path,
    row: dict[str, Any],
    attempt_dir: Path,
    snapshot_path: Path,
    snapshot_sha256: str,
    batch_path: Path,
    port: int,
) -> list[str]:
    command = FAST._sim_command(
        task_name=row["task_key"],
        mode="base_test",
        batch_path=batch_path,
        port=port,
        record_video_every_n=1,
        step_log_every_n=100,
    )
    evaluator_index = command.index(str(FAST.SIM_EVALUATOR))
    recovery_arguments = [
        str(RECOVERY_RUNNER),
        "--recovery-suite",
        str(suite_path.resolve()),
        "--recovery-scenario",
        row["scenario_id"],
        "--recovery-output-dir",
        str(attempt_dir),
        "--start-snapshot",
        str(snapshot_path),
        "--expected-start-snapshot-sha256",
        snapshot_sha256,
        "--per-episode-output",
    ]
    if row["protocol"] == "failure_start":
        recovery_arguments.append("--restore-only")
    command[evaluator_index : evaluator_index + 1] = recovery_arguments
    return command


def _attempt_directories(scenario_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in scenario_dir.glob("attempt-*")
        if path.is_dir() and path.name[len("attempt-") :].isdigit()
    )


def _next_attempt_directory(scenario_dir: Path) -> Path:
    indices = [
        int(path.name[len("attempt-") :])
        for path in _attempt_directories(scenario_dir)
    ]
    return scenario_dir / f"attempt-{max(indices, default=0) + 1:04d}"


def _nontrigger_audit(
    suite: dict[str, Any], row: dict[str, Any], trial_dir: Path, result: dict[str, Any]
) -> dict[str, Any] | None:
    summary = result.get("hrvla_recovery")
    if not isinstance(summary, dict) or summary.get("triggered") is not False:
        return None
    core = {
        "schema_version": 1,
        "status": "behavioral_failure_event_not_reached",
        "claim_boundary": "locked online failure was not delivered; oracle trial fails",
        "suite_sha256": canonical_sha256(suite),
        "task_id": row["task_id"],
        "scenario_id": row["scenario_id"],
        "episode_seed": int(result["episode_seed"]),
        "restore_audit_sha256": summary.get("restore_audit_sha256"),
        "episode_result_sha256": _file_sha256(trial_dir / "episode.json"),
    }
    report = {**core, "audit_sha256": canonical_sha256(core)}
    _write_json_atomic(trial_dir / "runtime-nontrigger-audit.json", report)
    return report


def _normalize_trial(
    lock: dict[str, Any],
    suite: dict[str, Any],
    capture_manifest: dict[str, Any],
    row: dict[str, Any],
    trial_index: int,
    trial_dir: Path,
) -> dict[str, Any]:
    scenario_id = row["scenario_id"]
    result_path = trial_dir / "episode.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("episode result must be a JSON object")
    expected_seed = int(lock["rollout_seeds"][scenario_id][trial_index])
    if int(result.get("episode_seed", -1)) != expected_seed:
        raise ValueError("oracle episode seed differs from the frozen trial")
    if int(result.get("episode_index", -1)) != trial_index:
        raise ValueError("oracle episode index differs from the frozen trial")
    if result.get("failure_reason") in {
        "interrupted",
        "process_error",
        "sim_error",
        "sim_stopped",
        "unknown",
    }:
        raise ValueError("oracle episode ended with an infrastructure failure")

    capture = _capture_row(capture_manifest, scenario_id)
    restore = json.loads((trial_dir / "restore-audit.json").read_text(encoding="utf-8"))
    start_prefix = "failure" if row["protocol"] == "failure_start" else "initial"
    validate_restore_audit(
        restore,
        expected_snapshot_sha256=capture[f"{start_prefix}_snapshot_sha256"],
        expected_task_id=row["task_id"],
        expected_event_id=(scenario_id if row["protocol"] == "failure_start" else "initial"),
        expected_simulator_revision=lock["independence"]["isaac_lab_revision"],
        expected_policy_rollout_seed=expected_seed,
    )
    evidence: dict[str, Any]
    success = bool(result.get("success"))
    failure_reason = str(result.get("failure_reason"))
    if row["protocol"] == "failure_start":
        evidence = audit_failure_start_trial(
            suite,
            scenario_id,
            trial_dir,
            result_path,
            expected_snapshot_sha256=capture["failure_snapshot_sha256"],
        )
        _write_json_atomic(trial_dir / "failure-start-trial-audit.json", evidence)
        protocol_evidence = {
            "failure_start_trial_audit_sha256": evidence["audit_sha256"]
        }
    else:
        try:
            evidence = audit_recovery_runtime_trace(
                suite, scenario_id, trial_dir, result_path
            )
            _write_json_atomic(trial_dir / "runtime-audit.json", evidence)
        except ValueError:
            evidence = _nontrigger_audit(suite, row, trial_dir, result)
            if evidence is None:
                raise
            success = False
            failure_reason = "failure_not_injected"
        protocol_evidence = {"runtime_audit_sha256": evidence["audit_sha256"]}

    independence = lock["independence"]
    normalized = {
        "schema_version": 1,
        "oracle_id": lock["oracle_id"],
        "suite_id": suite["suite_id"],
        "suite_sha256": canonical_sha256(suite),
        "capture_manifest_sha256": capture_manifest["manifest_sha256"],
        "task_id": row["task_id"],
        "scenario_id": scenario_id,
        "protocol": row["protocol"],
        "trial_index": int(trial_index),
        "rollout_seed": expected_seed,
        "model_revision": independence["model_revision"],
        "source_revision": independence["source_revision"],
        "isaac_lab_revision": independence["isaac_lab_revision"],
        "sonic_revision": independence["sonic_revision"],
        "initial_snapshot_sha256": capture["initial_snapshot_sha256"],
        "failure_snapshot_sha256": capture["failure_snapshot_sha256"],
        "start_state_restore_audit_sha256": restore["audit_sha256"],
        "success": success,
        "failure_reason": failure_reason,
        "upstream_episode_success": bool(result.get("success")),
        "upstream_failure_reason": str(result.get("failure_reason")),
        "episode_result_sha256": _file_sha256(result_path),
        "video_recorded": bool(result.get("video_recorded")),
        "video_path": str(result.get("video_path", "")),
        "evidence_directory": str(trial_dir),
        **protocol_evidence,
    }
    if lock["protocol"].get("record_all_trials") is True and (
        not normalized["video_recorded"] or not normalized["video_path"]
    ):
        raise ValueError("oracle trial lacks its locked video evidence")
    _write_json_atomic(trial_dir / "oracle-record.json", normalized)
    return normalized


def _discover_trial(
    lock: dict[str, Any],
    suite: dict[str, Any],
    capture_manifest: dict[str, Any],
    row: dict[str, Any],
    trial_index: int,
    scenario_dir: Path,
) -> dict[str, Any] | None:
    for attempt in reversed(_attempt_directories(scenario_dir)):
        trial_dir = attempt / f"trial-{trial_index:04d}"
        if not (trial_dir / "episode.json").is_file():
            continue
        try:
            return _normalize_trial(
                lock, suite, capture_manifest, row, trial_index, trial_dir
            )
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            continue
    return None


def _scenario_report(
    lock: dict[str, Any],
    suite: dict[str, Any],
    capture_manifest: dict[str, Any],
    row: dict[str, Any],
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scenario_dir = output_root / row["scenario_id"]
    records = []
    missing = []
    for trial_index in range(int(lock["protocol"]["required_trials"])):
        record = _discover_trial(
            lock, suite, capture_manifest, row, trial_index, scenario_dir
        )
        if record is None:
            missing.append(trial_index)
        else:
            records.append(record)
    if missing:
        report = {
            "schema_version": 1,
            "status": "incomplete",
            "scenario_id": row["scenario_id"],
            "trials_valid": len(records),
            "trials_required": int(lock["protocol"]["required_trials"]),
            "missing_trial_indices": missing,
            "admitted": False,
        }
    else:
        report = evaluate_oracle_trials(
            lock, suite, capture_manifest, row["scenario_id"], records
        )
    _write_json_atomic(scenario_dir / "oracle-report.json", report)
    return report, records


def _resource_blockers(max_idle_gpu_mib: int) -> list[str]:
    blockers = []
    processes = subprocess.run(
        ["ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True
    ).stdout
    for line in processes.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and int(fields[0]) != os.getpid():
            if "run_humanoidarena_baseline_matrix_fast.py" in fields[1]:
                blockers.append(f"external matrix is active at PID {fields[0]}")
            if "run_humanoidarena_recovery_admission.py" in fields[1]:
                blockers.append(f"runtime admission is active at PID {fields[0]}")
    gpu_used = FAST._gpu_used_mib()
    if gpu_used > max_idle_gpu_mib:
        blockers.append(f"compute GPU already uses {gpu_used} MiB (limit {max_idle_gpu_mib})")
    return blockers


def _progress(
    lock: dict[str, Any],
    suite: dict[str, Any],
    capture_manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    reports = []
    for row in rows:
        report, _records = _scenario_report(
            lock, suite, capture_manifest, row, output_root
        )
        reports.append(
            {
                "scenario_id": row["scenario_id"],
                "status": report["status"],
                "admitted": bool(report.get("admitted")),
                "trials": int(report.get("trials", report.get("trials_valid", 0))),
                "successes": int(report.get("successes", 0)),
            }
        )
    admitted = sum(report["admitted"] for report in reports)
    return {
        "schema_version": 1,
        "status": "oracle_complete" if admitted == len(rows) else "oracle_incomplete_or_rejected",
        "claim_boundary": "scenario recoverability admission only; not internal method scores",
        "suite_sha256": canonical_sha256(suite),
        "capture_manifest_sha256": capture_manifest["manifest_sha256"],
        "scenarios_admitted": admitted,
        "scenarios_required": len(rows),
        "scenarios": reports,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        default=ROOT / "benchmark/suites/hrvla_recovery_v0.json",
    )
    parser.add_argument(
        "--oracle-lock",
        type=Path,
        default=ROOT / "config/humanoidarena-recovery-oracle.lock.json",
    )
    parser.add_argument(
        "--capture-manifest",
        type=Path,
        default=ROOT
        / "_artifacts/HumanoidArena/recovery-admission/capture-manifest.json",
    )
    parser.add_argument(
        "--capture-runtime-root",
        type=Path,
        default=ROOT / "_artifacts/HumanoidArena/recovery-admission/runtime",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "_artifacts/HumanoidArena/recovery-admission/oracle",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=ROOT / "_artifacts/HumanoidArena/models",
    )
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--server-port", type=int, default=18444)
    parser.add_argument("--max-idle-gpu-mib", type=int, default=1024)
    args = parser.parse_args()

    suite = load_json(args.suite.resolve())
    lock = load_json(args.oracle_lock.resolve())
    capture_manifest = load_json(args.capture_manifest.resolve())
    validate_oracle_lock(lock, suite)
    if capture_manifest.get("status") != "runtime_and_snapshots_complete_oracle_admission_pending":
        raise ValueError("capture manifest is not complete")
    if capture_manifest.get("suite_sha256") != canonical_sha256(suite):
        raise ValueError("capture manifest suite hash differs")
    rows = _scenario_rows(suite)
    if args.scenario:
        selected = set(args.scenario)
        rows = [row for row in rows if row["scenario_id"] in selected]
        missing = selected - {row["scenario_id"] for row in rows}
        if missing:
            raise ValueError(f"unknown recovery scenarios: {sorted(missing)}")

    model_root = args.model_root.resolve()
    runtime_root = args.capture_runtime_root.resolve()
    output_root = args.output_root.resolve()
    FAST._validate_runtime(model_root)
    blockers = _resource_blockers(args.max_idle_gpu_mib)
    if blockers:
        for blocker in blockers:
            print(f"[recovery-oracle] blocked: {blocker}", file=sys.stderr)
        return 2
    output_root.mkdir(parents=True, exist_ok=True)

    with FAST.Telemetry(output_root / "hardware-telemetry.jsonl"):
        task_keys = list(dict.fromkeys(row["task_key"] for row in rows))
        for task_key in task_keys:
            task_rows = [row for row in rows if row["task_key"] == task_key]
            task_pending = []
            for row in task_rows:
                report, records = _scenario_report(
                    lock, suite, capture_manifest, row, output_root
                )
                if report["status"] == "incomplete":
                    task_pending.append((row, {record["trial_index"] for record in records}))
            if not task_pending:
                continue
            model_path = model_root / FAST.TASKS[task_key]["model"]
            log_root = output_root / "driver-logs" / task_key
            log_root.mkdir(parents=True, exist_ok=True)
            with (log_root / "server.log").open("a", encoding="utf-8", buffering=1) as log:
                server = subprocess.Popen(
                    FAST._server_command(model_path, args.server_port),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT,
                    env=FAST._server_env(24, 2, 32),
                    start_new_session=True,
                )
                try:
                    FAST._wait_for_server(args.server_port, 600.0)
                    for row, completed in task_pending:
                        scenario_id = row["scenario_id"]
                        scenario_dir = output_root / scenario_id
                        attempt_dir = _next_attempt_directory(scenario_dir)
                        attempt_dir.mkdir(parents=True, exist_ok=False)
                        snapshot_path, snapshot_sha = _start_snapshot(
                            capture_manifest, row, runtime_root
                        )
                        jobs = []
                        for trial_index, rollout_seed in enumerate(
                            lock["rollout_seeds"][scenario_id]
                        ):
                            if trial_index in completed:
                                continue
                            trial_dir = attempt_dir / f"trial-{trial_index:04d}"
                            jobs.append(
                                _episode_job(
                                    task_key,
                                    trial_dir,
                                    model_path,
                                    trial_index,
                                    rollout_seed,
                                )
                            )
                        batch_path = attempt_dir / "batch.json"
                        _write_json_atomic(batch_path, {"episodes": jobs})
                        command = build_oracle_command(
                            suite_path=args.suite,
                            row=row,
                            attempt_dir=attempt_dir,
                            snapshot_path=snapshot_path,
                            snapshot_sha256=snapshot_sha,
                            batch_path=batch_path,
                            port=args.server_port,
                        )
                        print(
                            f"[recovery-oracle] start {scenario_id} missing={len(jobs)}",
                            flush=True,
                        )
                        with (log_root / f"{scenario_id}.log").open(
                            "a", encoding="utf-8", buffering=1
                        ) as sim_log:
                            simulation = subprocess.Popen(
                                command,
                                stdout=sim_log,
                                stderr=subprocess.STDOUT,
                                cwd=FAST.ISAACLAB_ROOT,
                                env=FAST._sim_env(),
                                start_new_session=True,
                            )
                            try:
                                returncode = simulation.wait()
                            except KeyboardInterrupt:
                                FAST._terminate_group(simulation)
                                raise
                        if returncode != 0:
                            print(
                                f"[recovery-oracle] simulator status={returncode} for {scenario_id}",
                                file=sys.stderr,
                            )
                        _scenario_report(lock, suite, capture_manifest, row, output_root)
                        _write_json_atomic(
                            output_root / "progress.json",
                            _progress(lock, suite, capture_manifest, rows, output_root),
                        )
                finally:
                    FAST._terminate_group(server, timeout=10)

    progress = _progress(lock, suite, capture_manifest, rows, output_root)
    _write_json_atomic(output_root / "progress.json", progress)
    print(
        f"[recovery-oracle] admitted={progress['scenarios_admitted']}/"
        f"{progress['scenarios_required']}",
        flush=True,
    )
    return 0 if progress["scenarios_admitted"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
