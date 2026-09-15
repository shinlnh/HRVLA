#!/usr/bin/env python3
"""Run the HumanoidArena PI0.5+SONIC matrix with task-shared model servers.

The released evaluator restarts the 9.35 GB PI0.5 server for every
``(task, mode, seed)`` cell.  This driver keeps one exact CPU policy server
alive for all four modes and three seeds of a task, and runs every mode as one
persistent Isaac Sim batch.  It changes orchestration only: checkpoint,
precision, episode seeds, simulator configuration, controller, and horizons
remain locked to the released baseline.

Finished per-episode JSON files are resumable evidence.  A restarted run only
submits missing repeat IDs and atomically rebuilds a cell summary after all 20
repeats are present.  Video is sampled because encoding every episode does not
affect a metric and needlessly slows the claim-bearing matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HUMANOIDARENA_ROOT = ROOT / "_vendor" / "HumanoidArena"
ISAACLAB_PROJECT = HUMANOIDARENA_ROOT / "isaaclab_twist2_g1"
ISAACLAB_ROOT = ROOT / "_vendor" / "IsaacLab-v2.2.0"
SIM_PYTHON = ISAACLAB_ROOT / ".venv" / "bin" / "python"
POLICY_PYTHON = HUMANOIDARENA_ROOT / "lerobot" / ".venv" / "bin" / "python"
SIM_EVALUATOR = (
    ISAACLAB_PROJECT / "script" / "eval_scripts" / "sonic_pi05" / "sim_eval_vla.py"
)
SERVER = ROOT / "scripts" / "serve_humanoidarena_vla_low_memory.py"
SONIC_ROOT = (
    HUMANOIDARENA_ROOT / "GR00T-WholeBodyControl" / "gear_sonic_deploy" / "policy" / "release"
)

MODES = ("base_test", "semantic", "vision", "execution")
SEEDS = (0, 1, 2)
REPEATS = 20
MODEL_REVISION = "da13e072902840e2682afde360b763f1edb76d32"
SOURCE_REVISION = "68479287a784a69be9ce6ad739311d2f11f75ef9"
ISAACLAB_REVISION = "46dff135f44683f031edf346e544fcfd8456b2bb"

TASKS: dict[str, dict[str, Any]] = {
    "boxing": {
        "task_id": "Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody",
        "config_stem": "boxing_sonic_test.yaml",
        "model": "pi/HSI_boxing/pi05_sonic_boxing_0529/100000/pretrained_model",
        "max_steps": 900,
    },
    "doubledesk": {
        "task_id": "Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody",
        "config_stem": "doubledesk_sonic_test.yaml",
        "model": "pi/HOI_double_desk/pi05_sonic_doubledesk_0529/100000/pretrained_model",
        "max_steps": 2000,
    },
    "football": {
        "task_id": "Isaac-Move-Football-Single-G129-Dex3-Wholebody",
        "config_stem": "football_single_sonic_test.yaml",
        "model": "pi/HOI_football/pi05_sonic_football_0529/100000/pretrained_model",
        "max_steps": 2000,
    },
    "open_door": {
        "task_id": "Isaac-Move-Open-Door-G129-Dex3-Wholebody",
        "config_stem": "open_door_sonic_test.yaml",
        "model": "pi/HSI_open_door/pi05_sonic_opendoor_0529/100000/pretrained_model",
        "max_steps": 1800,
    },
    "pp_box": {
        "task_id": "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby",
        "config_stem": "pp_box_sonic_test.yaml",
        "model": "pi/HOI_pp_box/pi05_sonic_ppbox_0529/100000/pretrained_model",
        "max_steps": 1450,
    },
    "sit_sofa": {
        "task_id": "Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody",
        "config_stem": "sit_sofa_sonic_test.yaml",
        "model": "pi/HSI_sit_sofa/pi05_sonic_sitsofa_0529/100000/pretrained_model",
        "max_steps": 2000,
    },
    "vision_navi": {
        "task_id": "Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody",
        "config_stem": "vision_navi_sonic_test.yaml",
        "model": "pi/HSI_vision_navi/pi05_sonic_visionnavi_0529/100000/pretrained_model",
        "max_steps": 1800,
    },
}


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_runtime(model_root: Path) -> None:
    required = [SIM_PYTHON, POLICY_PYTHON, SIM_EVALUATOR, SERVER]
    required.extend((SONIC_ROOT / name) for name in ("model_encoder.onnx", "model_decoder.onnx"))
    required.extend(model_root / spec["model"] / "model.safetensors" for spec in TASKS.values())
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing baseline runtime inputs:\n" + "\n".join(missing))
    if _git_revision(HUMANOIDARENA_ROOT) != SOURCE_REVISION:
        raise RuntimeError("HumanoidArena source revision differs from the locked release")
    if _git_revision(ISAACLAB_ROOT) != ISAACLAB_REVISION:
        raise RuntimeError("Isaac Lab revision differs from the locked release")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _derive_episode_seed(task_name: str, group_seed: int, repeat_idx: int) -> int:
    payload = f"{task_name}|{int(group_seed)}|{int(repeat_idx)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[-4:], "big") & 0x7FFFFFFF


def _sanitize_label(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def _model_label(model_path: Path) -> str:
    return _sanitize_label(f"{model_path.parent.parent.name}__{model_path.parent.name}")


def _valid_episode_rows(cell_dir: Path, *, seed: int, repeats: int) -> list[dict[str, Any]]:
    by_repeat: dict[int, dict[str, Any]] = {}
    for path in sorted((cell_dir / "episodes").glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        repeat = int(row.get("repeat_idx", -1))
        reason = str(row.get("failure_reason", ""))
        if (
            int(row.get("seed", -1)) == seed
            and repeat in range(repeats)
            and reason not in {"interrupted", "process_error", "sim_error"}
            and int(row.get("returncode", 0)) == 0
        ):
            by_repeat[repeat] = row
    return [by_repeat[index] for index in sorted(by_repeat)]


def pending_repeat_ids(cell_dir: Path, *, seed: int, repeats: int) -> list[int]:
    completed = {
        int(row["repeat_idx"])
        for row in _valid_episode_rows(cell_dir, seed=seed, repeats=repeats)
    }
    return [repeat for repeat in range(repeats) if repeat not in completed]


def _mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def _gpu_used_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def _wait_for_capacity(min_available_gib: float, max_idle_gpu_mib: int) -> None:
    while _mem_available_gib() < min_available_gib or _gpu_used_mib() > max_idle_gpu_mib:
        print(
            "[fast-matrix] waiting for capacity "
            f"available_ram={_mem_available_gib():.1f}GiB gpu_used={_gpu_used_mib()}MiB",
            flush=True,
        )
        time.sleep(30)


def _server_env(cpu_threads: int, interop_threads: int, compile_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "CARB_APP_PATH", "EXP_PATH", "PYTHONHOME"):
        env.pop(key, None)
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": str(cpu_threads),
            "MKL_NUM_THREADS": str(cpu_threads),
            "OPENBLAS_NUM_THREADS": str(cpu_threads),
            "NUMEXPR_NUM_THREADS": str(cpu_threads),
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
            "OMP_WAIT_POLICY": "ACTIVE",
            "KMP_BLOCKTIME": "1",
            "MALLOC_ARENA_MAX": "4",
            "TORCHINDUCTOR_COMPILE_THREADS": str(compile_threads),
            "HRVLA_TORCH_INTEROP_THREADS": str(interop_threads),
            "HRVLA_VLA_DEBUG_LOGGING": "0",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    current_path = env.get("PATH", "")
    env["PATH"] = str(POLICY_PYTHON.parent) + (os.pathsep + current_path if current_path else "")
    return env


def _sim_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ISAACLAB_PROJECT),
            "ISAAC_PATH": str(
                ISAACLAB_ROOT / ".venv" / "lib" / "python3.11" / "site-packages" / "isaacsim"
            ),
            "LD_LIBRARY_PATH": str(ROOT / "_vendor" / "cyclonedds" / "install" / "lib"),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "LEROBOT_VLA_RECORD_OUTPUTS": "0",
        }
    )
    env.pop("CARB_APP_PATH", None)
    env.pop("EXP_PATH", None)
    return env


def _server_command(model_path: Path, port: int) -> list[str]:
    return [
        str(POLICY_PYTHON),
        "-u",
        str(SERVER),
        "--policy-path",
        str(model_path),
        "--device",
        "cpu",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def _post_reset(port: int, timeout: float) -> None:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/reset",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"PI0.5 server readiness returned HTTP {response.status}")


def _wait_for_server(port: int, timeout: float) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            _post_reset(port, 2.0)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"PI0.5 server did not become ready: {last_error}")


def _terminate_group(process: subprocess.Popen[Any], timeout: float = 30.0) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=timeout)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class Telemetry:
    def __init__(self, output_path: Path, interval: float = 5.0):
        self.output_path = output_path
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.previous_cpu: tuple[int, int] | None = None

    @staticmethod
    def _cpu_totals() -> tuple[int, int]:
        cpu_line = Path("/proc/stat").read_text().splitlines()[0]
        values = [int(value) for value in cpu_line.split()[1:]]
        total = sum(values)
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return total, idle

    def _sample(self) -> dict[str, Any]:
        cpu_total, cpu_idle = self._cpu_totals()
        cpu_busy = None
        if self.previous_cpu is not None:
            total_delta = cpu_total - self.previous_cpu[0]
            idle_delta = cpu_idle - self.previous_cpu[1]
            if total_delta > 0:
                cpu_busy = 100.0 * (1.0 - idle_delta / total_delta)
        self.previous_cpu = (cpu_total, cpu_idle)
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,"
                "power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().split(",")
        return {
            "timestamp": time.time(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cpu_busy_percent": cpu_busy,
            "memory_available_gib": _mem_available_gib(),
            "gpu_utilization_percent": float(gpu[0]),
            "gpu_memory_utilization_percent": float(gpu[1]),
            "gpu_memory_used_mib": float(gpu[2]),
            "gpu_power_w": float(gpu[3]),
            "gpu_temperature_c": float(gpu[4]),
        }

    def _run(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        while not self.stop_event.is_set():
            try:
                row = self._sample()
                with self.output_path.open("a", encoding="utf-8", buffering=1) as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
            except Exception as exc:
                print(f"[fast-matrix] telemetry warning: {exc}", flush=True)
            self.stop_event.wait(self.interval)

    def __enter__(self) -> Telemetry:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.interval + 2)


def _build_jobs(
    output_root: Path,
    *,
    task_name: str,
    mode: str,
    seeds: list[int],
    repeats: int,
    model_label: str,
    model_path: Path,
    max_steps: int,
) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
    jobs: list[dict[str, Any]] = []
    pending: dict[int, list[int]] = {}
    episode_index = 0
    task_id = str(TASKS[task_name]["task_id"])
    for seed in seeds:
        cell_dir = output_root / mode / task_name / f"seed-{seed}"
        pending[seed] = pending_repeat_ids(cell_dir, seed=seed, repeats=repeats)
        for repeat_idx in pending[seed]:
            stem = f"{model_label}__seed_{seed}__repeat_{repeat_idx}__episode_{episode_index}"
            jobs.append(
                {
                    "seed": seed,
                    "repeat_idx": repeat_idx,
                    "episode_seed": _derive_episode_seed(task_id, seed, repeat_idx),
                    "episode_index": episode_index,
                    "result_json": str(cell_dir / "episodes" / f"{stem}.json"),
                    "success_video_dir": str(cell_dir / "videos" / "success"),
                    "failure_video_dir": str(cell_dir / "videos" / "failure"),
                    "recording_save_dir": str(cell_dir / "recordings"),
                    "model_label": model_label,
                    "eval_model_path": str(model_path),
                    "max_steps": max_steps,
                    "video_fps": 30,
                    "post_termination_record_steps": 10,
                }
            )
            episode_index += 1
    return jobs, pending


def _sim_command(
    *,
    task_name: str,
    mode: str,
    batch_path: Path,
    port: int,
    record_video_every_n: int,
    step_log_every_n: int,
) -> list[str]:
    spec = TASKS[task_name]
    config_path = ISAACLAB_PROJECT / "tasks" / "common_test_config" / mode / spec["config_stem"]
    return [
        str(SIM_PYTHON),
        "-u",
        str(SIM_EVALUATOR),
        "--task",
        str(spec["task_id"]),
        "--env_config_yaml",
        str(config_path),
        "--seed",
        "0",
        "--max_steps",
        str(spec["max_steps"]),
        "--sonic_encoder_path",
        str(SONIC_ROOT / "model_encoder.onnx"),
        "--sonic_decoder_path",
        str(SONIC_ROOT / "model_decoder.onnx"),
        "--sonic_vla_root_rot6d_layout",
        "row",
        "--sonic_vla_root_max_delta_deg",
        "26",
        "--model_path",
        str(SONIC_ROOT / "model_encoder.onnx"),
        "--lerobot_server_url",
        f"http://127.0.0.1:{port}",
        "--lerobot_server_timeout",
        "900",
        "--robot_type",
        "unitree_g1_refpose_v3_1",
        "--record_video_every_n",
        str(record_video_every_n),
        "--step_log_every_n",
        str(step_log_every_n),
        "--recording_save_dir",
        str(batch_path.parent / "recordings"),
        "--episode_batch_json",
        str(batch_path),
        "--device",
        "cuda:0",
        "--enable_cameras",
        "--headless",
    ]


def _finalize_cell(
    cell_dir: Path,
    *,
    seed: int,
    repeats: int,
    model_path: Path,
    log_path: Path,
) -> bool:
    rows = _valid_episode_rows(cell_dir, seed=seed, repeats=repeats)
    if len(rows) != repeats:
        return False
    finalized = []
    for row in rows:
        row = dict(row)
        row["model_path"] = str(model_path)
        row["returncode"] = 0
        row["log_path"] = str(log_path)
        row.setdefault("vla_trace_path", "")
        finalized.append(row)
    _write_jsonl_atomic(cell_dir / "summary.jsonl", finalized)
    successes = sum(bool(row.get("success")) for row in finalized)
    reasons = Counter(str(row.get("failure_reason") or "unknown") for row in finalized)
    _write_json_atomic(
        cell_dir / "summary.json",
        {
            "episodes": repeats,
            "successes": successes,
            "failures": repeats - successes,
            "success_rate": successes / repeats,
            "result_reason_counts": dict(sorted(reasons.items())),
        },
    )
    return True


def _matrix_progress(
    output_root: Path, *, tasks: list[str], modes: list[str], seeds: list[int], repeats: int
) -> dict[str, Any]:
    cells_complete = 0
    episodes_observed = 0
    for task_name in tasks:
        for mode in modes:
            for seed in seeds:
                cell_dir = output_root / mode / task_name / f"seed-{seed}"
                rows = _valid_episode_rows(cell_dir, seed=seed, repeats=repeats)
                episodes_observed += len(rows)
                cells_complete += int(
                    len(rows) == repeats
                    and (cell_dir / "summary.jsonl").is_file()
                    and len((cell_dir / "summary.jsonl").read_text().splitlines()) == repeats
                )
    cells_total = len(tasks) * len(modes) * len(seeds)
    return {
        "schema_version": 2,
        "runner": "task-shared-server",
        "model_revision": MODEL_REVISION,
        "source_revision": SOURCE_REVISION,
        "isaaclab_revision": ISAACLAB_REVISION,
        "cells_total": cells_total,
        "cells_completed": cells_complete,
        "episodes_expected": cells_total * repeats,
        "episodes_observed": episodes_observed,
        "complete": cells_complete == cells_total,
        "updated_at": time.time(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "_artifacts/HumanoidArena/paper-baselines/pi05-sonic",
    )
    parser.add_argument("--model-root", type=Path, default=ROOT / "_artifacts/HumanoidArena/models")
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--record-video-every-n", type=int, default=10)
    parser.add_argument("--step-log-every-n", type=int, default=250)
    parser.add_argument("--cpu-threads", type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument("--interop-threads", type=int, default=2)
    parser.add_argument("--compile-threads", type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument("--server-port", type=int, default=18443)
    parser.add_argument("--server-ready-timeout", type=float, default=600.0)
    parser.add_argument("--min-start-ram-gib", type=float, default=20.0)
    parser.add_argument("--min-runtime-ram-gib", type=float, default=3.0)
    parser.add_argument("--max-idle-gpu-mib", type=int, default=2048)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    available_cpus = len(os.sched_getaffinity(0))
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.cpu_threads not in range(1, available_cpus + 1):
        parser.error(f"--cpu-threads must be in 1..{available_cpus}")
    if args.interop_threads < 1 or args.compile_threads < 1:
        parser.error("thread counts must be positive")
    if args.record_video_every_n < 0:
        parser.error("--record-video-every-n cannot be negative")

    output_root = args.output_root.resolve()
    model_root = args.model_root.resolve()
    _validate_runtime(model_root)

    if args.dry_run:
        for task_name in args.tasks:
            spec = TASKS[task_name]
            model_path = model_root / spec["model"]
            model_label = _model_label(model_path)
            task_jobs = 0
            for mode in args.modes:
                jobs, _ = _build_jobs(
                    output_root,
                    task_name=task_name,
                    mode=mode,
                    seeds=args.seeds,
                    repeats=args.repeats,
                    model_label=model_label,
                    model_path=model_path,
                    max_steps=int(spec["max_steps"]),
                )
                task_jobs += len(jobs)
                print(f"[fast-matrix] dry-run {task_name}/{mode} missing={len(jobs)}")
            if task_jobs:
                print(" ".join(_server_command(model_path, args.server_port)))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.json"
    telemetry_path = output_root / "hardware-telemetry.jsonl"
    _write_json_atomic(
        progress_path,
        _matrix_progress(
            output_root,
            tasks=args.tasks,
            modes=args.modes,
            seeds=args.seeds,
            repeats=args.repeats,
        ),
    )

    with Telemetry(telemetry_path):
        for task_name in args.tasks:
            spec = TASKS[task_name]
            model_path = model_root / spec["model"]
            model_label = _model_label(model_path)
            task_has_work = any(
                pending_repeat_ids(
                    output_root / mode / task_name / f"seed-{seed}",
                    seed=seed,
                    repeats=args.repeats,
                )
                for mode in args.modes
                for seed in args.seeds
            )
            if not task_has_work:
                print(f"[fast-matrix] task complete, skip server: {task_name}", flush=True)
                continue

            _wait_for_capacity(args.min_start_ram_gib, args.max_idle_gpu_mib)
            log_root = output_root / "driver-logs" / task_name
            log_root.mkdir(parents=True, exist_ok=True)
            server_log_path = log_root / "server.log"
            with server_log_path.open("a", encoding="utf-8", buffering=1) as server_log:
                server = subprocess.Popen(
                    _server_command(model_path, args.server_port),
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT,
                    env=_server_env(args.cpu_threads, args.interop_threads, args.compile_threads),
                    start_new_session=True,
                )
                try:
                    print(f"[fast-matrix] loading shared policy task={task_name}", flush=True)
                    _wait_for_server(args.server_port, args.server_ready_timeout)
                    for mode in args.modes:
                        jobs, pending = _build_jobs(
                            output_root,
                            task_name=task_name,
                            mode=mode,
                            seeds=args.seeds,
                            repeats=args.repeats,
                            model_label=model_label,
                            model_path=model_path,
                            max_steps=int(spec["max_steps"]),
                        )
                        if not jobs:
                            print(f"[fast-matrix] mode complete, skip: {task_name}/{mode}")
                            continue
                        for seed in args.seeds:
                            cell_dir = output_root / mode / task_name / f"seed-{seed}"
                            cell_dir.mkdir(parents=True, exist_ok=True)
                            _write_json_atomic(
                                cell_dir / "runtime.json",
                                {
                                    "schema_version": 2,
                                    "orchestration": "task-shared-server",
                                    "cpu_threads": args.cpu_threads,
                                    "interop_threads": args.interop_threads,
                                    "compile_threads": args.compile_threads,
                                    "record_video_every_n": args.record_video_every_n,
                                    "pending_repeat_ids_at_start": pending[seed],
                                    "started_at": time.time(),
                                },
                            )
                        batch_path = log_root / f"{mode}-batch.json"
                        _write_json_atomic(batch_path, {"episodes": jobs})
                        sim_log_path = log_root / f"{mode}.log"
                        print(
                            f"[fast-matrix] start task={task_name} mode={mode} "
                            f"missing_episodes={len(jobs)}",
                            flush=True,
                        )
                        with sim_log_path.open("a", encoding="utf-8", buffering=1) as sim_log:
                            simulation = subprocess.Popen(
                                _sim_command(
                                    task_name=task_name,
                                    mode=mode,
                                    batch_path=batch_path,
                                    port=args.server_port,
                                    record_video_every_n=args.record_video_every_n,
                                    step_log_every_n=args.step_log_every_n,
                                ),
                                stdout=sim_log,
                                stderr=subprocess.STDOUT,
                                cwd=ISAACLAB_ROOT,
                                env=_sim_env(),
                                start_new_session=True,
                            )
                            try:
                                while simulation.poll() is None:
                                    if _mem_available_gib() < args.min_runtime_ram_gib:
                                        sim_log.write("[fast-matrix] RAM safety stop\n")
                                        _terminate_group(simulation)
                                        break
                                    time.sleep(10)
                            except KeyboardInterrupt:
                                _terminate_group(simulation)
                                raise
                        if simulation.returncode != 0:
                            raise RuntimeError(
                                f"simulator failed task={task_name} mode={mode} "
                                f"returncode={simulation.returncode}; see {sim_log_path}"
                            )
                        for seed in args.seeds:
                            _finalize_cell(
                                output_root / mode / task_name / f"seed-{seed}",
                                seed=seed,
                                repeats=args.repeats,
                                model_path=model_path,
                                log_path=sim_log_path,
                            )
                        _write_json_atomic(
                            progress_path,
                            _matrix_progress(
                                output_root,
                                tasks=args.tasks,
                                modes=args.modes,
                                seeds=args.seeds,
                                repeats=args.repeats,
                            ),
                        )
                finally:
                    _terminate_group(server, timeout=10)

    progress = _matrix_progress(
        output_root,
        tasks=args.tasks,
        modes=args.modes,
        seeds=args.seeds,
        repeats=args.repeats,
    )
    _write_json_atomic(progress_path, progress)
    print(
        f"[fast-matrix] complete={progress['complete']} "
        f"cells={progress['cells_completed']}/{progress['cells_total']} "
        f"episodes={progress['episodes_observed']}/{progress['episodes_expected']}",
        flush=True,
    )
    return 0 if progress["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
