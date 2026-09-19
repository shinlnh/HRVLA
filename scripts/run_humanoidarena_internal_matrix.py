#!/usr/bin/env python3
"""Run the frozen five-method HumanoidArena matrix without mixing evidence.

The driver is intentionally gated by an admitted recovery suite, immutable
checkpoint manifests, frozen plan hashes, and the recorded GR00T/Isaac memory
probe.  It keeps one GR00T server alive for every checkpoint family/seed and
batches all rollout seeds of one task/scenario into one Isaac Sim process.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.internal_matrix import (  # noqa: E402
    audit_internal_records,
    group_plan_cells,
    load_checkpoint_lock,
    normalize_internal_record,
    validate_ready_checkpoint_lock,
)
from hrvla_bench.internal_protocol import INTERNAL_METHODS  # noqa: E402
from hrvla_bench.humanoidarena_release import verify_release_manifest  # noqa: E402
from hrvla_bench.hidden_final_gate import validate_hidden_final_gate  # noqa: E402
from hrvla_bench.plan import canonical_sha256, load_json  # noqa: E402
from hrvla_bench.recovery_restore import (  # noqa: E402
    audit_failure_start_trial,
    validate_restore_audit,
)
from hrvla_bench.recovery_runtime_audit import (  # noqa: E402
    audit_recovery_runtime_trace,
)
from hrvla_bench.score import validate_record  # noqa: E402


FAST_RUNNER = ROOT / "scripts/run_humanoidarena_baseline_matrix_fast.py"
INTERNAL_EPISODE_RUNNER = ROOT / "scripts/run_humanoidarena_internal_episode.py"
GR00T_SERVER = ROOT / "scripts/serve_humanoidarena_gr00t.py"
GR00T_PYTHON = ROOT / "_vendor/Isaac-GR00T/.venv/bin/python"
INFRASTRUCTURE_FAILURES = {
    "interrupted",
    "process_error",
    "sim_error",
    "sim_stopped",
    "unknown",
}


def _load_fast_runner():
    spec = importlib.util.spec_from_file_location("hrvla_internal_fast", FAST_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import orchestration primitives: {FAST_RUNNER}")
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


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_inside_root(relative: str, label: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository root") from exc
    return path


def verify_checkpoint_files(lock: dict[str, Any]) -> None:
    """Verify the local cache against the immutable published manifest lock."""

    for relative, expected in lock["method_runtime_signatures"].items():
        path = _resolve_inside_root(relative, "method runtime signature path")
        if not path.is_file() or _file_sha256(path) != expected:
            raise ValueError(f"method runtime signature differs: {path}")
    checked: set[tuple[str, str]] = set()
    for method_id in INTERNAL_METHODS:
        for seed in lock["checkpoints"][method_id]:
            row = lock["checkpoints"][method_id][seed]
            identity = (row["checkpoint_id"], row["manifest_sha256"])
            if identity in checked:
                continue
            model_path = _resolve_inside_root(row["path"], "checkpoint path")
            manifest_path = _resolve_inside_root(row["manifest_path"], "manifest path")
            if not model_path.is_dir():
                raise FileNotFoundError(model_path)
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            manifest = load_json(manifest_path)
            if manifest.get("manifest_sha256") != row["manifest_sha256"]:
                raise ValueError(f"checkpoint manifest hash differs: {manifest_path}")
            if manifest.get("root") != row["path"]:
                raise ValueError(f"checkpoint manifest root differs: {manifest_path}")
            errors = verify_release_manifest(manifest, ROOT)
            if errors:
                raise ValueError(
                    f"checkpoint artifact differs: {manifest_path}: " + "; ".join(errors)
                )
            checked.add(identity)


def _server_command(checkpoint: dict[str, Any], *, port: int, device: str) -> list[str]:
    return [
        str(GR00T_PYTHON),
        "-u",
        str(GR00T_SERVER),
        "--model-path",
        str(_resolve_inside_root(checkpoint["path"], "checkpoint path")),
        "--device",
        device,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--denoising-steps",
        "4",
    ]


def _server_env(device: str, cpu_threads: int) -> dict[str, str]:
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "CARB_APP_PATH", "EXP_PATH", "PYTHONHOME"):
        environment.pop(key, None)
    threads = cpu_threads if device == "cpu" else 2
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }
    )
    path = environment.get("PATH", "")
    environment["PATH"] = str(GR00T_PYTHON.parent) + (os.pathsep + path if path else "")
    return environment


def _capture_row(capture_manifest: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    for row in capture_manifest["scenarios"]:
        if row["scenario_id"] == scenario_id:
            return row
    raise ValueError(f"capture manifest has no scenario {scenario_id}")


def _snapshot_for_cell(
    suite: dict[str, Any],
    capture_manifest: dict[str, Any],
    capture_runtime_root: Path,
    cell: dict[str, Any],
) -> tuple[Path, str]:
    task = next(row for row in suite["tasks"] if row["id"] == cell["task_id"])
    if cell["protocol"] == "nominal":
        capture = next(
            row
            for row in capture_manifest["scenarios"]
            if row["task_id"] == cell["task_id"]
        )
        relative = capture["initial_snapshot_path"]
        expected = task["admission"]["initial_snapshot_sha256"]
    else:
        scenario = next(
            row for row in task["scenarios"] if row["id"] == cell["scenario_id"]
        )
        capture = _capture_row(capture_manifest, cell["scenario_id"])
        prefix = "failure" if cell["protocol"] == "failure_start" else "initial"
        relative = capture[f"{prefix}_snapshot_path"]
        expected = (
            scenario["admission"]["failure_snapshot_sha256"]
            if prefix == "failure"
            else task["admission"]["initial_snapshot_sha256"]
        )
    path = (capture_runtime_root / relative).resolve()
    try:
        path.relative_to(capture_runtime_root.resolve())
    except ValueError as exc:
        raise ValueError("capture snapshot path escapes the runtime root") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, expected


def _episode_job(
    task_key: str,
    trial_dir: Path,
    checkpoint_path: Path,
    episode: dict[str, Any],
    episode_index: int,
) -> dict[str, Any]:
    spec = FAST.TASKS[task_key]
    return {
        "seed": int(episode["training_seed"]),
        "repeat_idx": int(episode_index),
        "episode_seed": int(episode["rollout_seed"]),
        "episode_index": int(episode_index),
        "result_json": str(trial_dir / "episode.json"),
        "success_video_dir": str(trial_dir / "videos" / "success"),
        "failure_video_dir": str(trial_dir / "videos" / "failure"),
        "recording_save_dir": str(trial_dir / "recordings"),
        "model_label": FAST._model_label(checkpoint_path),
        "eval_model_path": str(checkpoint_path),
        "max_steps": int(spec["max_steps"]),
        "video_fps": 30,
        "post_termination_record_steps": 10,
    }


def build_internal_command(
    *,
    method_id: str,
    task_key: str,
    scenario_id: str | None,
    suite_path: Path,
    method_programs_path: Path,
    attempt_dir: Path,
    snapshot_path: Path,
    snapshot_sha256: str,
    batch_path: Path,
    port: int,
    record_video_every_n: int,
) -> list[str]:
    command = FAST._sim_command(
        task_name=task_key,
        mode="base_test",
        batch_path=batch_path,
        port=port,
        record_video_every_n=record_video_every_n,
        step_log_every_n=100,
    )
    evaluator_index = command.index(str(FAST.SIM_EVALUATOR))
    replacement = [
        str(INTERNAL_EPISODE_RUNNER),
        "--internal-method",
        method_id,
        "--method-programs",
        str(method_programs_path.resolve()),
        "--suite",
        str(suite_path.resolve()),
        "--internal-output-dir",
        str(attempt_dir),
        "--start-snapshot",
        str(snapshot_path),
        "--expected-start-snapshot-sha256",
        snapshot_sha256,
        "--per-episode-output",
    ]
    if scenario_id is not None:
        replacement.extend(["--recovery-scenario", scenario_id])
    command[evaluator_index : evaluator_index + 1] = replacement
    return command


def _attempt_directories(cell_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in cell_dir.glob("attempt-*")
        if path.is_dir() and path.name.removeprefix("attempt-").isdigit()
    )


def _next_attempt_directory(cell_dir: Path) -> Path:
    indices = [int(path.name.removeprefix("attempt-")) for path in _attempt_directories(cell_dir)]
    return cell_dir / f"attempt-{max(indices, default=0) + 1:04d}"


def _record_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["task_id"],
        row["scenario_id"],
        row["training_seed"],
        row["rollout_seed"],
    )


def _discover_records(method_root: Path, plan: dict[str, Any], method_id: str) -> list[dict[str, Any]]:
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in sorted(method_root.glob("**/record.json")):
        try:
            row = load_json(path)
            validate_record(row)
            if row["method_id"] != method_id or row["plan_sha256"] != plan["plan_sha256"]:
                continue
            key = _record_key(row)
            if key not in rows:
                rows[key] = row
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return list(rows.values())


def _audit_trial(
    suite: dict[str, Any],
    cell: dict[str, Any],
    trial_dir: Path,
    snapshot_sha256: str,
    simulator_revision: str,
) -> str:
    result_path = trial_dir / "episode.json"
    result = load_json(result_path)
    if str(result.get("failure_reason", "unknown")) in INFRASTRUCTURE_FAILURES:
        raise ValueError("upstream episode ended with an infrastructure failure")
    episode = next(
        row for row in cell["episodes"] if row["rollout_seed"] == result.get("episode_seed")
    )
    if cell["protocol"] == "nominal":
        audit = load_json(trial_dir / "restore-audit.json")
        validate_restore_audit(
            audit,
            expected_snapshot_sha256=snapshot_sha256,
            expected_task_id=cell["task_id"],
            expected_event_id="initial",
            expected_simulator_revision=simulator_revision,
            expected_policy_rollout_seed=episode["rollout_seed"],
        )
        return audit["audit_sha256"]
    if cell["protocol"] == "failure_start":
        audit = audit_failure_start_trial(
            suite,
            cell["scenario_id"],
            trial_dir,
            result_path,
            expected_snapshot_sha256=snapshot_sha256,
        )
        _write_json_atomic(trial_dir / "failure-start-trial-audit.json", audit)
        return audit["audit_sha256"]
    audit = audit_recovery_runtime_trace(suite, cell["scenario_id"], trial_dir, result_path)
    _write_json_atomic(trial_dir / "runtime-audit.json", audit)
    return audit["audit_sha256"]


def _resource_blockers(max_idle_gpu_mib: int) -> list[str]:
    blockers = []
    patterns = (
        "run_humanoidarena_baseline_matrix_fast.py",
        "run_humanoidarena_recovery_admission.py",
        "run_humanoidarena_recovery_oracle.py",
        "run_humanoidarena_gr00t_adaptation.py",
        "run_humanoidarena_gr00t_validation.py",
        "run_humanoidarena_gr00t_hidden_eval.py",
    )
    processes = subprocess.run(
        ["ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True
    ).stdout
    for line in processes.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or int(fields[0]) == os.getpid():
            continue
        for pattern in patterns:
            if pattern in fields[1]:
                blockers.append(f"{pattern} is active at PID {fields[0]}")
                break
    gpu_used = FAST._gpu_used_mib()
    if gpu_used > max_idle_gpu_mib:
        blockers.append(f"compute GPU already uses {gpu_used} MiB (limit {max_idle_gpu_mib})")
    return blockers


def _progress(
    split: str,
    plan: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    by_method = {
        method_id: len(_discover_records(output_root / split / method_id, plan, method_id))
        for method_id in INTERNAL_METHODS
    }
    expected_per_method = len(plan["episodes"])
    observed = sum(by_method.values())
    expected = expected_per_method * len(INTERNAL_METHODS)
    return {
        "schema_version": 1,
        "split": split,
        "plan_sha256": plan["plan_sha256"],
        "methods": by_method,
        "records_observed": observed,
        "records_expected": expected,
        "complete": observed == expected,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _run_cell(
    *,
    split: str,
    method_id: str,
    plan: dict[str, Any],
    cell: dict[str, Any],
    checkpoint: dict[str, Any],
    suite: dict[str, Any],
    suite_path: Path,
    capture_manifest: dict[str, Any],
    capture_runtime_root: Path,
    method_programs_path: Path,
    method_program_sha256: str,
    output_root: Path,
    port: int,
    record_video_every_n: int,
    controller_id: str,
    simulator_revision: str,
) -> int:
    method_root = output_root / split / method_id
    existing = {_record_key(row) for row in _discover_records(method_root, plan, method_id)}
    pending = [episode for episode in cell["episodes"] if _record_key({**episode}) not in existing]
    if not pending:
        return 0
    task = next(row for row in suite["tasks"] if row["id"] == cell["task_id"])
    task_key = task["humanoidarena_task_key"]
    if task_key not in FAST.TASKS:
        raise ValueError(f"unknown HumanoidArena task key: {task_key}")
    snapshot_path, snapshot_sha256 = _snapshot_for_cell(
        suite, capture_manifest, capture_runtime_root, cell
    )
    cell_dir = (
        method_root
        / cell["task_id"]
        / cell["scenario_id"]
        / f"training-seed-{cell['training_seed']}"
    )
    attempt_dir = _next_attempt_directory(cell_dir)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = _resolve_inside_root(checkpoint["path"], "checkpoint path")
    jobs = []
    for index, episode in enumerate(pending):
        trial_dir = attempt_dir / f"trial-{index:04d}"
        jobs.append(_episode_job(task_key, trial_dir, checkpoint_path, episode, index))
    batch_path = attempt_dir / "batch.json"
    _write_json_atomic(batch_path, {"episodes": jobs})
    command = build_internal_command(
        method_id=method_id,
        task_key=task_key,
        scenario_id=None if cell["protocol"] == "nominal" else cell["scenario_id"],
        suite_path=suite_path,
        method_programs_path=method_programs_path,
        attempt_dir=attempt_dir,
        snapshot_path=snapshot_path,
        snapshot_sha256=snapshot_sha256,
        batch_path=batch_path,
        port=port,
        record_video_every_n=record_video_every_n,
    )
    print(
        f"[internal-matrix] start {split}/{method_id}/{cell['scenario_id']}/"
        f"seed-{cell['training_seed']} missing={len(pending)}",
        flush=True,
    )
    log_path = attempt_dir / "simulator.log"
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        simulation = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=FAST.ISAACLAB_ROOT,
            env=FAST._sim_env(),
            start_new_session=True,
        )
        try:
            while simulation.poll() is None:
                if FAST._mem_available_gib() < 2.5:
                    log.write("[internal-matrix] RAM safety stop\n")
                    FAST._terminate_group(simulation)
                    break
                time.sleep(10)
        except KeyboardInterrupt:
            FAST._terminate_group(simulation)
            raise
    errors = int(simulation.returncode != 0)
    for index, episode in enumerate(pending):
        trial_dir = attempt_dir / f"trial-{index:04d}"
        try:
            evidence_sha256 = _audit_trial(
                suite, cell, trial_dir, snapshot_sha256, simulator_revision
            )
            result = load_json(trial_dir / "episode.json")
            record = normalize_internal_record(
                plan,
                episode,
                result,
                method_id=method_id,
                checkpoint=checkpoint,
                run_id=f"humanoidarena-internal-{split}-{method_id}",
                episode_id=f"{method_id}::{episode['episode_key']}",
                controller_id=controller_id,
                simulator_revision=simulator_revision,
                method_program_sha256=method_program_sha256,
            )
            record["diagnostics"]["runtime_evidence_sha256"] = evidence_sha256
            _write_json_atomic(trial_dir / "record.json", record)
        except Exception as exc:
            errors += 1
            _write_json_atomic(
                trial_dir / "normalization-error.json",
                {
                    "schema_version": 1,
                    "episode_key": episode["episode_key"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "claim_boundary": "invalid trial; never included in paper evidence",
                },
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "validation", "hidden_final"), default="development")
    parser.add_argument(
        "--plans-root",
        type=Path,
        default=ROOT / "_artifacts/HumanoidArena/internal-benchmark/plans",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=ROOT / "_artifacts/HumanoidArena/recovery-admission/hrvla_recovery_v0.admitted.json",
    )
    parser.add_argument(
        "--capture-manifest",
        type=Path,
        default=ROOT / "_artifacts/HumanoidArena/recovery-admission/capture-manifest.json",
    )
    parser.add_argument(
        "--capture-runtime-root",
        type=Path,
        default=ROOT / "_artifacts/HumanoidArena/recovery-admission/runtime",
    )
    parser.add_argument(
        "--checkpoint-lock",
        type=Path,
        default=ROOT / "config/humanoidarena-internal-checkpoints.lock.json",
    )
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=ROOT / "config/humanoidarena-internal-protocol.lock.json",
    )
    parser.add_argument(
        "--method-programs",
        type=Path,
        default=ROOT / "benchmark/humanoidarena_method_programs.json",
    )
    parser.add_argument(
        "--hidden-final-gate",
        type=Path,
        default=ROOT / "config/humanoidarena-hidden-final.lock.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "_artifacts/HumanoidArena/internal-benchmark/runs",
    )
    parser.add_argument("--server-port", type=int, default=18445)
    parser.add_argument("--cpu-threads", type=int, default=24)
    parser.add_argument("--record-video-every-n", type=int, default=13)
    parser.add_argument("--max-idle-gpu-mib", type=int, default=1024)
    parser.add_argument("--server-ready-timeout", type=float, default=1200.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    def request_shutdown(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    if args.cpu_threads not in range(1, len(os.sched_getaffinity(0)) + 1):
        parser.error("--cpu-threads is outside the available CPU affinity")
    if args.record_video_every_n < 1:
        parser.error("--record-video-every-n must be positive")

    method_programs = load_json(args.method_programs.resolve())
    protocol_lock = load_json(args.protocol_lock.resolve())
    checkpoint_lock = load_checkpoint_lock(args.checkpoint_lock.resolve())
    method_program_sha256 = canonical_sha256(method_programs)
    protocol_sha256 = canonical_sha256(protocol_lock)
    validate_ready_checkpoint_lock(
        checkpoint_lock,
        method_program_sha256=method_program_sha256,
        internal_protocol_sha256=protocol_sha256,
    )
    verify_checkpoint_files(checkpoint_lock)
    suite_path = args.suite.resolve()
    suite = load_json(suite_path)
    capture_manifest = load_json(args.capture_manifest.resolve())
    plan = load_json(args.plans_root.resolve() / f"{args.split}.plan.json")
    if plan["suite_sha256"] != canonical_sha256(suite):
        raise ValueError("plan suite hash differs from the admitted suite")
    if set(plan["methods"]) != set(INTERNAL_METHODS):
        raise ValueError("plan does not contain exactly the five internal methods")
    if args.split == "hidden_final":
        validate_hidden_final_gate(
            load_json(args.hidden_final_gate.resolve()),
            validation_plan=load_json(args.plans_root.resolve() / "validation.plan.json"),
            protocol=protocol_lock,
            checkpoint_lock=checkpoint_lock,
            method_programs=method_programs,
            validation_root=args.output_root.resolve() / "validation",
        )
    cells = group_plan_cells(plan)
    device = checkpoint_lock["policy_device"]
    output_root = args.output_root.resolve()

    grouped: dict[tuple[int, str], list[str]] = defaultdict(list)
    for method_id in INTERNAL_METHODS:
        for training_seed in checkpoint_lock["training_seeds"]:
            checkpoint = checkpoint_lock["checkpoints"][method_id][str(training_seed)]
            grouped[(training_seed, checkpoint["checkpoint_id"])].append(method_id)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "split": args.split,
                    "plan_sha256": plan["plan_sha256"],
                    "cells_per_method": len(cells),
                    "episodes_per_method": len(plan["episodes"]),
                    "checkpoint_server_loads": len(grouped),
                    "policy_device": device,
                },
                sort_keys=True,
            )
        )
        return 0

    blockers = _resource_blockers(args.max_idle_gpu_mib)
    if blockers:
        for blocker in blockers:
            print(f"[internal-matrix] blocked: {blocker}", file=sys.stderr)
        return 3
    FAST._validate_runtime((ROOT / "_artifacts/HumanoidArena/models").resolve())
    output_root.mkdir(parents=True, exist_ok=True)
    errors = 0
    with FAST.Telemetry(output_root / args.split / "hardware-telemetry.jsonl"):
        for (training_seed, checkpoint_id), method_ids in grouped.items():
            checkpoint = checkpoint_lock["checkpoints"][method_ids[0]][str(training_seed)]
            pending_methods = [
                method_id
                for method_id in method_ids
                if len(_discover_records(output_root / args.split / method_id, plan, method_id))
                < len(plan["episodes"])
            ]
            if not pending_methods:
                continue
            FAST._wait_for_capacity(20.0, args.max_idle_gpu_mib)
            log_root = output_root / args.split / "driver-logs" / f"seed-{training_seed}"
            log_root.mkdir(parents=True, exist_ok=True)
            server_log_path = log_root / f"{checkpoint_id}.server.log"
            with server_log_path.open("a", encoding="utf-8", buffering=1) as server_log:
                server = subprocess.Popen(
                    _server_command(checkpoint, port=args.server_port, device=device),
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT,
                    env=_server_env(device, args.cpu_threads),
                    start_new_session=True,
                )
                try:
                    FAST._wait_for_server(args.server_port, args.server_ready_timeout)
                    for method_id in pending_methods:
                        method_checkpoint = checkpoint_lock["checkpoints"][method_id][str(training_seed)]
                        for cell in cells:
                            if cell["training_seed"] != training_seed:
                                continue
                            errors += _run_cell(
                                split=args.split,
                                method_id=method_id,
                                plan=plan,
                                cell=cell,
                                checkpoint=method_checkpoint,
                                suite=suite,
                                suite_path=suite_path,
                                capture_manifest=capture_manifest,
                                capture_runtime_root=args.capture_runtime_root.resolve(),
                                method_programs_path=args.method_programs,
                                method_program_sha256=method_program_sha256,
                                output_root=output_root,
                                port=args.server_port,
                                record_video_every_n=args.record_video_every_n,
                                controller_id=checkpoint_lock["controller_id"],
                                simulator_revision=checkpoint_lock["simulator_revision"],
                            )
                            _write_json_atomic(
                                output_root / args.split / "progress.json",
                                _progress(args.split, plan, output_root),
                            )
                finally:
                    FAST._terminate_group(server, timeout=20)

    for method_id in INTERNAL_METHODS:
        records = _discover_records(output_root / args.split / method_id, plan, method_id)
        _write_jsonl_atomic(output_root / args.split / method_id / "records.jsonl", records)
        if len(records) == len(plan["episodes"]):
            audit = audit_internal_records(plan, method_id, records)
            _write_json_atomic(output_root / args.split / method_id / "audit.json", audit)
    progress = _progress(args.split, plan, output_root)
    _write_json_atomic(output_root / args.split / "progress.json", progress)
    print(json.dumps(progress, sort_keys=True), flush=True)
    return 0 if progress["complete"] and errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
