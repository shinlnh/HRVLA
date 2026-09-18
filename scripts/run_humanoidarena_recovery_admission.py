#!/usr/bin/env python3
"""Capture and audit one live recovery-injector trace for every locked scenario.

This is the runtime-evidence stage, not the independent 20/20 oracle stage.  It
uses the released task policy only to reach each semantic boundary, records one
representative video, and never changes suite admission fields automatically.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.plan import canonical_sha256, load_json  # noqa: E402
from hrvla_bench.recovery_runtime_audit import audit_recovery_runtime_trace  # noqa: E402


FAST_RUNNER_PATH = ROOT / "scripts/run_humanoidarena_baseline_matrix_fast.py"
RECOVERY_RUNNER = ROOT / "scripts/run_humanoidarena_recovery_episode.py"


def _load_fast_runner():
    spec = importlib.util.spec_from_file_location("hrvla_fast_matrix", FAST_RUNNER_PATH)
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


def _task_rows(suite: dict[str, Any], scenario_filter: set[str]) -> list[dict[str, Any]]:
    rows = []
    for task in suite["tasks"]:
        task_key = str(task["humanoidarena_task_key"])
        if task_key not in FAST.TASKS:
            raise ValueError(f"suite task key is absent from the locked runner: {task_key}")
        for scenario in task["scenarios"]:
            if not scenario_filter or scenario["id"] in scenario_filter:
                rows.append(
                    {
                        "task_id": task["id"],
                        "task_key": task_key,
                        "scenario_id": scenario["id"],
                        "protocol": scenario["protocol"],
                    }
                )
    found = {row["scenario_id"] for row in rows}
    missing = scenario_filter - found
    if missing:
        raise ValueError(f"unknown recovery scenarios: {sorted(missing)}")
    return rows


def _episode_job(
    task_key: str,
    output_dir: Path,
    model_path: Path,
    episode_seed: int,
) -> dict[str, Any]:
    spec = FAST.TASKS[task_key]
    return {
        "seed": 0,
        "repeat_idx": 0,
        "episode_seed": int(episode_seed),
        "episode_index": 0,
        "result_json": str(output_dir / "episode.json"),
        "success_video_dir": str(output_dir / "videos" / "success"),
        "failure_video_dir": str(output_dir / "videos" / "failure"),
        "recording_save_dir": str(output_dir / "recordings"),
        "model_label": FAST._model_label(model_path),
        "eval_model_path": str(model_path),
        "max_steps": int(spec["max_steps"]),
        "video_fps": 30,
        "post_termination_record_steps": 10,
    }


def build_recovery_command(
    *,
    task_key: str,
    scenario_id: str,
    protocol: str,
    output_dir: Path,
    batch_path: Path,
    port: int,
    suite_path: Path = ROOT / "benchmark/suites/hrvla_recovery_v0.json",
) -> list[str]:
    command = FAST._sim_command(
        task_name=task_key,
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
        scenario_id,
        "--recovery-output-dir",
        str(output_dir),
        "--capture-initial-snapshot",
    ]
    if protocol == "failure_start":
        recovery_arguments.append("--capture-failure-snapshot")
    command[evaluator_index : evaluator_index + 1] = recovery_arguments
    return command


def _resource_blockers(max_idle_gpu_mib: int) -> list[str]:
    blockers = []
    processes = subprocess.run(
        ["ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True
    ).stdout
    current_pid = os.getpid()
    for line in processes.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or int(fields[0]) == current_pid:
            continue
        if "run_humanoidarena_baseline_matrix_fast.py" in fields[1]:
            blockers.append(f"external matrix is active at PID {fields[0]}")
    gpu_used = FAST._gpu_used_mib()
    if gpu_used > max_idle_gpu_mib:
        blockers.append(f"compute GPU already uses {gpu_used} MiB (limit {max_idle_gpu_mib})")
    return blockers


def _attempt_directories(scenario_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in scenario_dir.glob("attempt-*")
        if path.is_dir() and path.name[len("attempt-") :].isdigit()
    )


def _validated_existing(
    suite: dict[str, Any],
    scenario_id: str,
    scenario_dir: Path,
    implementation_revision: str | None = None,
) -> tuple[dict[str, Any], Path] | None:
    for output_dir in reversed(_attempt_directories(scenario_dir)):
        result_path = output_dir / "episode.json"
        if not result_path.is_file():
            continue
        try:
            audit = audit_recovery_runtime_trace(
                suite, scenario_id, output_dir, result_path
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
        if (
            implementation_revision is not None
            and audit.get("implementation_revision") != implementation_revision
        ):
            continue
        return audit, output_dir
    return None


def _next_attempt_directory(scenario_dir: Path) -> Path:
    indices = [
        int(path.name[len("attempt-") :])
        for path in _attempt_directories(scenario_dir)
    ]
    return scenario_dir / f"attempt-{max(indices, default=0) + 1:04d}"


def _progress(
    suite: dict[str, Any],
    rows: list[dict[str, Any]],
    output_root: Path,
    implementation_revision: str | None = None,
) -> dict[str, Any]:
    scenario_rows = []
    for row in rows:
        scenario_dir = output_root / row["scenario_id"]
        validated = _validated_existing(
            suite, row["scenario_id"], scenario_dir, implementation_revision
        )
        audit = None if validated is None else validated[0]
        scenario_rows.append(
            {
                **row,
                "runtime_trace_validated": audit is not None,
                "audit_sha256": None if audit is None else audit["audit_sha256"],
                "evidence_directory": (
                    None
                    if validated is None
                    else str(validated[1].relative_to(output_root))
                ),
            }
        )
    complete = sum(item["runtime_trace_validated"] for item in scenario_rows)
    return {
        "schema_version": 1,
        "status": (
            "runtime_traces_complete_oracle_admission_pending"
            if complete == len(rows)
            else "runtime_trace_capture_incomplete"
        ),
        "claim_boundary": "runtime injector evidence only; no oracle or recoverability claim",
        "suite_sha256": canonical_sha256(suite),
        "runtime_traces_complete": complete,
        "runtime_traces_required": len(rows),
        "scenarios": scenario_rows,
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
        "--output-root",
        type=Path,
        default=ROOT / "_artifacts/HumanoidArena/recovery-admission/runtime",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=ROOT / "_artifacts/HumanoidArena/models",
    )
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--server-port", type=int, default=18444)
    parser.add_argument("--cpu-threads", type=int, default=FAST.DEFAULT_POLICY_THREADS)
    parser.add_argument("--interop-threads", type=int, default=2)
    parser.add_argument("--compile-threads", type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument("--max-idle-gpu-mib", type=int, default=1024)
    parser.add_argument("--server-ready-timeout", type=float, default=600.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    suite = load_json(args.suite.resolve())
    implementation_revision = FAST._git_revision(ROOT)
    rows = _task_rows(suite, set(args.scenario))
    output_root = args.output_root.resolve()
    model_root = args.model_root.resolve()
    episode_seed = int(suite["admission_capture"]["snapshot_seed"])
    FAST._validate_runtime(model_root)

    commands = []
    for row in rows:
        output_dir = output_root / row["scenario_id"] / "attempt-0001"
        model_path = model_root / FAST.TASKS[row["task_key"]]["model"]
        batch_path = output_dir / "batch.json"
        commands.append(
            build_recovery_command(
                task_key=row["task_key"],
                scenario_id=row["scenario_id"],
                protocol=row["protocol"],
                output_dir=output_dir,
                batch_path=batch_path,
                port=args.server_port,
                suite_path=args.suite,
            )
        )
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return 0

    blockers = _resource_blockers(args.max_idle_gpu_mib)
    if blockers:
        for blocker in blockers:
            print(f"[recovery-admission] blocked: {blocker}", file=sys.stderr)
        return 2

    output_root.mkdir(parents=True, exist_ok=True)
    failures = 0
    with FAST.Telemetry(output_root / "hardware-telemetry.jsonl"):
        task_keys = list(dict.fromkeys(row["task_key"] for row in rows))
        for task_key in task_keys:
            task_rows = [row for row in rows if row["task_key"] == task_key]
            pending = [
                row
                for row in task_rows
                if _validated_existing(
                    suite,
                    row["scenario_id"],
                    output_root / row["scenario_id"],
                    implementation_revision,
                )
                is None
            ]
            if not pending:
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
                    env=FAST._server_env(
                        args.cpu_threads, args.interop_threads, args.compile_threads
                    ),
                    start_new_session=True,
                )
                try:
                    FAST._wait_for_server(args.server_port, args.server_ready_timeout)
                    for row in pending:
                        scenario_id = row["scenario_id"]
                        scenario_dir = output_root / scenario_id
                        output_dir = _next_attempt_directory(scenario_dir)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        batch_path = output_dir / "batch.json"
                        _write_json_atomic(
                            batch_path,
                            {
                                "episodes": [
                                    _episode_job(
                                        task_key, output_dir, model_path, episode_seed
                                    )
                                ]
                            },
                        )
                        command = build_recovery_command(
                            task_key=task_key,
                            scenario_id=scenario_id,
                            protocol=row["protocol"],
                            output_dir=output_dir,
                            batch_path=batch_path,
                            port=args.server_port,
                            suite_path=args.suite,
                        )
                        print(
                            f"[recovery-admission] start {task_key}/{scenario_id}",
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
                        try:
                            if returncode != 0:
                                raise RuntimeError(f"simulator exited with status {returncode}")
                            audit = audit_recovery_runtime_trace(
                                suite,
                                scenario_id,
                                output_dir,
                                output_dir / "episode.json",
                            )
                            _write_json_atomic(output_dir / "runtime-audit.json", audit)
                        except Exception as exc:
                            failures += 1
                            _write_json_atomic(
                                output_dir / "runtime-audit-error.json",
                                {
                                    "schema_version": 1,
                                    "scenario_id": scenario_id,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                    "claim_boundary": "failed runtime evidence; not admitted",
                                },
                            )
                            print(
                                f"[recovery-admission] failed {scenario_id}: {exc}",
                                file=sys.stderr,
                                flush=True,
                            )
                        _write_json_atomic(
                            output_root / "progress.json",
                            _progress(
                                suite, rows, output_root, implementation_revision
                            ),
                        )
                finally:
                    FAST._terminate_group(server, timeout=10)

    progress = _progress(suite, rows, output_root, implementation_revision)
    _write_json_atomic(output_root / "progress.json", progress)
    print(
        f"[recovery-admission] runtime={progress['runtime_traces_complete']}/"
        f"{progress['runtime_traces_required']} oracle=0/{len(rows)}",
        flush=True,
    )
    return 0 if not failures and progress["runtime_traces_complete"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
