#!/usr/bin/env python3
"""Probe the exact GR00T + camera + SONIC runtime before freezing its device."""

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
import threading
import time
from typing import Any
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
FAST_PATH = ROOT / "scripts/run_humanoidarena_baseline_matrix_fast.py"
INTERNAL_PATH = ROOT / "scripts/run_humanoidarena_internal_matrix.py"
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.humanoidarena_release import git_revision, verify_release_manifest  # noqa: E402
from hrvla_bench.plan import load_json  # noqa: E402


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FAST = _load_script("hrvla_probe_fast", FAST_PATH)
INTERNAL = _load_script("hrvla_probe_internal", INTERNAL_PATH)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace coexistence evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _compute_vram_mib() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return sum(int(line.strip()) for line in result.stdout.splitlines() if line.strip())


def _gpu_sample() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [float(value.strip()) for value in result.stdout.strip().split(",")]
    return {
        "recorded_at_utc": _utc_now(),
        "gpu_utilization_percent": values[0],
        "gpu_memory_utilization_percent": values[1],
        "gpu_memory_used_mib": values[2],
        "gpu_power_w": values[3],
        "gpu_temperature_c": values[4],
        "compute_memory_used_mib": _compute_vram_mib(),
    }


class Telemetry:
    def __init__(self, path: Path, interval: float) -> None:
        self.path = path
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x", encoding="utf-8", buffering=1) as stream:
            while not self._stop.is_set():
                try:
                    sample = _gpu_sample()
                    self.samples.append(sample)
                    stream.write(json.dumps(sample, sort_keys=True) + "\n")
                except (OSError, subprocess.SubprocessError, ValueError) as error:
                    stream.write(json.dumps({"recorded_at_utc": _utc_now(), "error": str(error)}) + "\n")
                self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval + 5)

    @property
    def peak_compute_vram_mib(self) -> int:
        return max((int(row["compute_memory_used_mib"]) for row in self.samples if "compute_memory_used_mib" in row), default=0)


def _terminate_group(process: subprocess.Popen[Any] | None, timeout: float = 30.0) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=timeout)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _health(port: int) -> dict[str, Any]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=10) as response:
        return json.loads(response.read())


def build_probe_job(
    output_dir: Path, checkpoint: Path, *, steps: int, seed: int
) -> dict[str, Any]:
    return {
        "seed": seed,
        "repeat_idx": 0,
        "episode_seed": seed,
        "episode_index": 0,
        "result_json": str(output_dir / "episode.json"),
        "success_video_dir": str(output_dir / "videos/success"),
        "failure_video_dir": str(output_dir / "videos/failure"),
        "recording_save_dir": str(output_dir / "recordings"),
        "model_label": FAST._model_label(checkpoint),
        "eval_model_path": str(checkpoint),
        "max_steps": steps,
        "video_fps": 30,
        "post_termination_record_steps": 2,
    }


def _attempt(
    *,
    checkpoint: Path,
    device: str,
    output_dir: Path,
    port: int,
    steps: int,
    seed: int,
    telemetry_interval: float,
    ready_timeout: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True)
    batch = output_dir / "episode-batch.json"
    _write_json_once(batch, {"schema_version": 1, "jobs": [build_probe_job(output_dir, checkpoint, steps=steps, seed=seed)]})
    server_log_path = output_dir / "server.log"
    simulator_log_path = output_dir / "simulator.log"
    telemetry = Telemetry(output_dir / "hardware-telemetry.jsonl", telemetry_interval)
    server = None
    simulator = None
    error = None
    started_at = _utc_now()
    telemetry.start()
    try:
        with server_log_path.open("x", encoding="utf-8") as server_log:
            server = subprocess.Popen(
                INTERNAL._server_command(
                    {"path": checkpoint.relative_to(ROOT).as_posix()},
                    port=port,
                    device=device,
                ),
                cwd=ROOT,
                env=INTERNAL._server_env(device, 24),
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            FAST._wait_for_server(port, ready_timeout)
            command = FAST._sim_command(
                task_name="boxing",
                mode="base_test",
                batch_path=batch,
                port=port,
                record_video_every_n=1,
                step_log_every_n=max(1, steps),
            )
            with simulator_log_path.open("x", encoding="utf-8") as simulator_log:
                simulator = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=FAST._sim_env(),
                    stdout=simulator_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                )
                simulator_returncode = simulator.wait()
            health = _health(port)
            episode_path = output_dir / "episode.json"
            episode = json.loads(episode_path.read_text(encoding="utf-8")) if episode_path.is_file() else None
            if simulator_returncode != 0:
                raise RuntimeError(f"simulator exited {simulator_returncode}")
            if int(health.get("infer_count", 0)) < 1:
                raise RuntimeError("probe did not execute a real GR00T inference request")
            if not isinstance(episode, dict) or int(episode.get("returncode", 0)) != 0:
                raise RuntimeError("probe did not preserve a valid simulator episode record")
    except Exception as exception:  # evidence must survive GPU OOM and runtime failure
        error = f"{type(exception).__name__}: {exception}"
        health = None
        episode = None
        simulator_returncode = simulator.poll() if simulator is not None else None
    finally:
        _terminate_group(simulator)
        _terminate_group(server)
        telemetry.stop()
    return {
        "device": device,
        "status": "pass" if error is None else "fail",
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "peak_compute_vram_mib": telemetry.peak_compute_vram_mib,
        "telemetry_samples": len(telemetry.samples),
        "health": health,
        "simulator_returncode": simulator_returncode,
        "episode_result": episode,
        "error": error,
    }


def _blockers(max_idle_gpu_mib: int) -> list[str]:
    patterns = (
        "run_humanoidarena_baseline_matrix_fast.py",
        "run_humanoidarena_recovery_admission.py",
        "run_humanoidarena_recovery_oracle.py",
        "run_humanoidarena_gr00t_adaptation.py",
        "run_humanoidarena_gr00t_validation.py",
        "run_humanoidarena_gr00t_hidden_eval.py",
        "run_humanoidarena_rt_training.py",
        "run_humanoidarena_rt_validation.py",
        "run_humanoidarena_internal_matrix.py",
    )
    own_pid = os.getpid()
    processes = subprocess.run(
        ["ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True
    ).stdout
    blockers = []
    for line in processes.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and int(fields[0]) != own_pid and any(pattern in fields[1] for pattern in patterns):
            blockers.append(f"active benchmark process: {fields[0]} {fields[1]}")
    used = _compute_vram_mib()
    if used > max_idle_gpu_mib:
        blockers.append(f"pre-existing compute VRAM is {used} MiB, limit is {max_idle_gpu_mib} MiB")
    return blockers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "_artifacts/HumanoidArena/release/humanoidarena-v1/coexistence")
    parser.add_argument("--output", type=Path, default=ROOT / "_artifacts/HumanoidArena/release/humanoidarena-v1/coexistence-probe.json")
    parser.add_argument("--maximum-peak-compute-vram-mib", type=int, default=15500)
    parser.add_argument("--maximum-idle-compute-vram-mib", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--port", type=int, default=18446)
    parser.add_argument("--telemetry-interval", type=float, default=0.5)
    parser.add_argument("--server-ready-timeout", type=float, default=1200)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    checkpoint.relative_to(ROOT.resolve())
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    manifest = load_json(args.manifest.resolve())
    errors = verify_release_manifest(manifest, ROOT)
    if errors:
        raise ValueError("checkpoint release manifest differs: " + "; ".join(errors))
    if (ROOT / manifest["root"]).resolve() != checkpoint:
        raise ValueError("checkpoint path differs from the release manifest")
    manifest_sha256 = manifest["manifest_sha256"]
    if args.steps < 1 or args.telemetry_interval <= 0:
        raise ValueError("steps and telemetry interval must be positive")
    blockers = _blockers(args.maximum_idle_compute_vram_mib)
    if blockers:
        for blocker in blockers:
            print(f"blocked: {blocker}", file=sys.stderr)
        return 3
    if args.output.exists() or args.output_dir.exists():
        raise FileExistsError("coexistence probe evidence already exists; audit it before retrying")

    gpu = _attempt(
        checkpoint=checkpoint,
        device="cuda:0",
        output_dir=args.output_dir.resolve() / "gpu-attempt",
        port=args.port,
        steps=args.steps,
        seed=args.seed,
        telemetry_interval=args.telemetry_interval,
        ready_timeout=args.server_ready_timeout,
    )
    gpu_pass = (
        gpu["status"] == "pass"
        and 0 < gpu["peak_compute_vram_mib"] <= args.maximum_peak_compute_vram_mib
    )
    cpu = None
    if not gpu_pass:
        cpu = _attempt(
            checkpoint=checkpoint,
            device="cpu",
            output_dir=args.output_dir.resolve() / "cpu-fallback",
            port=args.port,
            steps=args.steps,
            seed=args.seed,
            telemetry_interval=args.telemetry_interval,
            ready_timeout=args.server_ready_timeout,
        )
        if cpu["status"] != "pass":
            result = {
                "schema_version": 1,
                "status": "failed",
                "policy_device": None,
                "measured_peak_compute_vram_mib": gpu["peak_compute_vram_mib"],
                "maximum_peak_compute_vram_mib": args.maximum_peak_compute_vram_mib,
                "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(),
                "checkpoint_manifest_sha256": manifest_sha256,
                "source_revision": git_revision(ROOT),
                "gpu_attempt": gpu,
                "cpu_fallback": cpu,
            }
            _write_json_once(args.output.resolve(), result)
            return 2
    result = {
        "schema_version": 1,
        "status": "passed" if gpu_pass else "failed_cpu_fallback",
        "policy_device": "cuda:0" if gpu_pass else "cpu",
        "measured_peak_compute_vram_mib": gpu["peak_compute_vram_mib"],
        "maximum_peak_compute_vram_mib": args.maximum_peak_compute_vram_mib,
        "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_manifest_sha256": manifest_sha256,
        "source_revision": git_revision(ROOT),
        "gpu_attempt": gpu,
        "cpu_fallback": cpu,
    }
    _write_json_once(args.output.resolve(), result)
    print(json.dumps({key: result[key] for key in ("status", "policy_device", "measured_peak_compute_vram_mib")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
