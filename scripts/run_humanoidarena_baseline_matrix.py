#!/usr/bin/env python3
"""Run the released HumanoidArena PI0.5+SONIC baseline matrix safely.

The runner deliberately executes one task/mode/seed cell at a time.  Each cell
uses HumanoidArena's persistent simulator for all repeats, keeps the 9.35 GB
PI0.5 policy on CPU, and reserves the single CUDA device for Isaac Sim and
SONIC.  Completed cells are detected from their machine-readable summaries so
an interrupted multi-day run can be resumed without repeating valid evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HUMANOIDARENA_ROOT = ROOT / "_vendor" / "HumanoidArena"
ISAACLAB_PROJECT = HUMANOIDARENA_ROOT / "isaaclab_twist2_g1"
ISAACLAB_ROOT = ROOT / "_vendor" / "IsaacLab-v2.2.0"
SIM_PYTHON = ISAACLAB_ROOT / ".venv" / "bin" / "python"
POLICY_PYTHON = HUMANOIDARENA_ROOT / "lerobot" / ".venv" / "bin" / "python"
EVALUATOR = ISAACLAB_PROJECT / "script" / "eval_scripts" / "sonic_pi05" / "eval_vla_suite.py"
SERVER = ROOT / "scripts" / "serve_humanoidarena_vla_low_memory.py"
SONIC_ROOT = (
    HUMANOIDARENA_ROOT / "GR00T-WholeBodyControl" / "gear_sonic_deploy" / "policy" / "release"
)

MODES = ("base_test", "semantic", "vision", "execution")
DEFAULT_SEEDS = (0, 1, 2)
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
        "model": ("pi/HOI_double_desk/pi05_sonic_doubledesk_0529/100000/pretrained_model"),
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
    required = [SIM_PYTHON, POLICY_PYTHON, EVALUATOR, SERVER]
    required.extend((SONIC_ROOT / name) for name in ("model_encoder.onnx", "model_decoder.onnx"))
    required.extend(model_root / spec["model"] / "model.safetensors" for spec in TASKS.values())
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing baseline runtime inputs:\n" + "\n".join(missing))
    if _git_revision(HUMANOIDARENA_ROOT) != SOURCE_REVISION:
        raise RuntimeError("HumanoidArena source revision differs from the locked release")
    if _git_revision(ISAACLAB_ROOT) != ISAACLAB_REVISION:
        raise RuntimeError("Isaac Lab revision differs from the locked release")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def cell_complete(cell_dir: Path, *, seed: int, repeats: int) -> bool:
    rows = _load_rows(cell_dir / "summary.jsonl")
    if len(rows) != repeats:
        return False
    repeat_ids = {int(row.get("repeat_idx", -1)) for row in rows}
    if repeat_ids != set(range(repeats)):
        return False
    return all(
        int(row.get("seed", -1)) == seed
        and int(row.get("returncode", -1)) == 0
        and row.get("failure_reason") != "process_error"
        for row in rows
    )


def _mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def _gpu_used_mib() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def _wait_for_capacity(min_available_gib: float, max_idle_gpu_mib: int) -> None:
    while True:
        available = _mem_available_gib()
        gpu_used = _gpu_used_mib()
        if available >= min_available_gib and gpu_used <= max_idle_gpu_mib:
            return
        print(
            "[matrix] waiting for capacity "
            f"available_ram={available:.1f}GiB gpu_used={gpu_used}MiB",
            flush=True,
        )
        time.sleep(30)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _command(
    *,
    task_name: str,
    mode: str,
    seed: int,
    repeats: int,
    model_root: Path,
    cell_dir: Path,
) -> list[str]:
    spec = TASKS[task_name]
    config_path = ISAACLAB_PROJECT / "tasks" / "common_test_config" / mode / spec["config_stem"]
    model_path = model_root / spec["model"]
    return [
        str(SIM_PYTHON),
        "-u",
        str(EVALUATOR),
        "--task",
        spec["task_id"],
        "--env_config_yaml",
        str(config_path),
        "--model-path",
        str(model_path),
        "--seed",
        str(seed),
        "--repeats_per_seed",
        str(repeats),
        "--persistent_sim",
        "1",
        "--max_steps",
        str(spec["max_steps"]),
        "--video_fps",
        "30",
        "--post_termination_record_steps",
        "10",
        "--record_video_every_n",
        "1",
        "--step_log_every_n",
        "100",
        "--robot_type",
        "unitree_g1_refpose_v3_1",
        "--sonic_encoder_path",
        str(SONIC_ROOT / "model_encoder.onnx"),
        "--sonic_decoder_path",
        str(SONIC_ROOT / "model_decoder.onnx"),
        "--sonic_vla_root_rot6d_layout",
        "row",
        "--sonic_vla_root_max_delta_deg",
        "26",
        "--results_dir",
        str(cell_dir),
        "--headless",
        "--isaac_device",
        "cuda:0",
        "--server_python",
        str(POLICY_PYTHON),
        "--server_script",
        str(SERVER),
        "--server_device",
        "cpu",
        "--server_host",
        "127.0.0.1",
        "--server_port",
        "18443",
        "--server_ready_timeout",
        "240",
        "--lerobot_server_timeout",
        "900",
    ]


def _runtime_env() -> dict[str, str]:
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
            "OMP_NUM_THREADS": "24",
            "MKL_NUM_THREADS": "24",
            "TORCHINDUCTOR_COMPILE_THREADS": "16",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }
    )
    env.pop("CARB_APP_PATH", None)
    env.pop("EXP_PATH", None)
    return env


def _terminate_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_cell(command: list[str], cell_dir: Path, min_runtime_ram_gib: float) -> int:
    cell_dir.mkdir(parents=True, exist_ok=True)
    log_path = cell_dir / "matrix-driver.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
            env=_runtime_env(),
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                available = _mem_available_gib()
                if available < min_runtime_ram_gib:
                    log.write(
                        f"[matrix] safety stop: available RAM {available:.2f} GiB "
                        f"< {min_runtime_ram_gib:.2f} GiB\n"
                    )
                    _terminate_group(process)
                    return 70
                time.sleep(10)
        except KeyboardInterrupt:
            _terminate_group(process)
            raise
        return int(process.returncode or 0)


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
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--min-start-ram-gib", type=float, default=20.0)
    parser.add_argument("--min-runtime-ram-gib", type=float, default=3.0)
    parser.add_argument("--max-idle-gpu-mib", type=int, default=2048)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if any(seed not in DEFAULT_SEEDS for seed in args.seeds):
        parser.error("paper baseline seeds must be selected from 0, 1, 2")

    model_root = args.model_root.resolve()
    output_root = args.output_root.resolve()
    _validate_runtime(model_root)
    cells = [
        (task_name, mode, seed)
        for task_name in args.tasks
        for mode in args.modes
        for seed in args.seeds
    ]
    progress_path = output_root / "progress.json"
    completed = 0
    failed = 0

    for index, (task_name, mode, seed) in enumerate(cells, start=1):
        cell_dir = output_root / mode / task_name / f"seed-{seed}"
        if cell_complete(cell_dir, seed=seed, repeats=args.repeats):
            completed += 1
            print(f"[matrix] resume skip {index}/{len(cells)} {mode}/{task_name}/seed-{seed}")
            continue

        command = _command(
            task_name=task_name,
            mode=mode,
            seed=seed,
            repeats=args.repeats,
            model_root=model_root,
            cell_dir=cell_dir,
        )
        if args.dry_run:
            print(" ".join(command))
            continue

        _wait_for_capacity(args.min_start_ram_gib, args.max_idle_gpu_mib)
        print(f"[matrix] start {index}/{len(cells)} {mode}/{task_name}/seed-{seed}", flush=True)
        returncode = _run_cell(command, cell_dir, args.min_runtime_ram_gib)
        valid = returncode == 0 and cell_complete(
            cell_dir,
            seed=seed,
            repeats=args.repeats,
        )
        completed += int(valid)
        failed += int(not valid)
        progress = {
            "schema_version": 1,
            "model_revision": MODEL_REVISION,
            "source_revision": SOURCE_REVISION,
            "isaaclab_revision": ISAACLAB_REVISION,
            "cells_total": len(cells),
            "cells_completed": completed,
            "cells_failed": failed,
            "episodes_expected": len(cells) * args.repeats,
            "episodes_completed": completed * args.repeats,
            "last_cell": {"task": task_name, "mode": mode, "seed": seed},
            "last_returncode": returncode,
            "updated_at": time.time(),
        }
        _write_json_atomic(progress_path, progress)
        if not valid:
            print(f"[matrix] cell failed validation: {mode}/{task_name}/seed-{seed}", flush=True)
            return 1

    if not args.dry_run:
        _write_json_atomic(
            progress_path,
            {
                "schema_version": 1,
                "model_revision": MODEL_REVISION,
                "source_revision": SOURCE_REVISION,
                "isaaclab_revision": ISAACLAB_REVISION,
                "cells_total": len(cells),
                "cells_completed": completed,
                "cells_failed": failed,
                "episodes_expected": len(cells) * args.repeats,
                "episodes_completed": completed * args.repeats,
                "complete": completed == len(cells) and failed == 0,
                "updated_at": time.time(),
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
