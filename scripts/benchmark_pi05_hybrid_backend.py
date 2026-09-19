#!/usr/bin/env python3
"""Compare the locked PI0.5 CPU backend with the opt-in hybrid backend.

This is a backend qualification pilot, not paper evidence.  It launches each
backend in a separate process, sends identical seeded requests, records full
action chunks, and evaluates latency, numerical drift, RAM, and CUDA memory.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "scripts" / "serve_humanoidarena_vla_low_memory.py"
BACKEND_SOURCE = ROOT / "src" / "hrvla_bench" / "pi05_hybrid_backend.py"
CUDA_INT8_SOURCE = ROOT / "src" / "hrvla_bench" / "pi05_cuda_int8_backend.py"
DEFAULT_POLICY = (
    ROOT
    / "_artifacts"
    / "HumanoidArena"
    / "models"
    / "pi"
    / "HOI_pp_box"
    / "pi05_sonic_ppbox_0529"
    / "100000"
    / "pretrained_model"
)
DEFAULT_OUTPUT = (
    ROOT
    / "_artifacts"
    / "HumanoidArena"
    / "backend-pilots"
    / "pi05-hybrid-qualification.json"
)
CANDIDATE_BACKENDS = ("hybrid_cuda_expert", "cuda_int8_weight_only")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (SERVER, BACKEND_SOURCE, CUDA_INT8_SOURCE):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _fixtures() -> list[dict]:
    height, width = 480, 640
    black = np.zeros((height, width, 3), dtype=np.uint8)
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    gradient = np.empty_like(black)
    gradient[..., 0] = x
    gradient[..., 1] = y
    gradient[..., 2] = np.bitwise_xor(x, y)
    yy, xx = np.indices((height, width))
    checker_value = (((xx // 32) + (yy // 32)) % 2 * 255).astype(np.uint8)
    checker = np.repeat(checker_value[..., None], 3, axis=2)
    states = (
        np.zeros(64, dtype=np.float32),
        np.linspace(-0.2, 0.2, 64, dtype=np.float32),
        (0.15 * np.sin(np.linspace(0.0, 4.0 * math.pi, 64))).astype(np.float32),
    )
    fixtures = []
    for index, (name, image, state) in enumerate(
        zip(("black-neutral", "gradient-ramp", "checker-sine"), (black, gradient, checker), states)
    ):
        fixtures.append(
            {
                "name": name,
                "seed": 20260919 + index,
                "image": image,
                "state": state,
                "image_sha256": _sha256_bytes(image.tobytes()),
                "state_sha256": _sha256_bytes(state.tobytes()),
            }
        )
    return fixtures


def _payload(fixture: dict) -> dict:
    image = fixture["image"]
    return {
        "observation": {
            "images": {
                "front": {
                    "shape": list(image.shape),
                    "dtype": str(image.dtype),
                    "data_b64": base64.b64encode(image.tobytes()).decode("ascii"),
                }
            },
            "state": fixture["state"].tolist(),
        },
        "robot_type": "unitree_g1_refpose_v3_1",
        "task": "HOI_pp_box",
        "return_chunk": True,
    }


def _post(port: int, path: str, payload: dict, timeout: float) -> tuple[dict, float]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    return body, time.perf_counter() - started


def _wait_for_server(process: subprocess.Popen, port: int, timeout: float) -> float:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"PI0.5 server exited with code {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return time.perf_counter() - started
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"PI0.5 server did not listen on port {port} within {timeout}s")


def _process_gpu_mib(pid: int) -> float:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            return float(fields[1])
    return 0.0


def _process_rss_mib(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    return 0.0


class ResourceMonitor:
    def __init__(self, pid: int):
        self.pid = pid
        self.peak_gpu_mib = 0.0
        self.peak_rss_mib = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.2):
            self.peak_gpu_mib = max(self.peak_gpu_mib, _process_gpu_mib(self.pid))
            self.peak_rss_mib = max(self.peak_rss_mib, _process_rss_mib(self.pid))

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        self._thread.join(timeout=5)


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _run_backend(
    backend: str,
    policy: Path,
    fixtures: list[dict],
    port: int,
    output_dir: Path,
    cpu_threads: int,
) -> dict:
    log_path = output_dir / f"{backend}.log"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "HRVLA_PI05_BACKEND": backend,
            "HRVLA_PI05_EXPERT_DEVICE": "cuda:0",
            "HRVLA_TORCH_INTEROP_THREADS": "2",
            "HRVLA_VLA_DEBUG_LOGGING": "0",
            "OMP_NUM_THREADS": str(cpu_threads),
        }
    )
    device = "cuda:0" if backend == "cuda_int8_weight_only" else "cpu"
    command = [
        str(ROOT / "_vendor" / "HumanoidArena" / "lerobot" / ".venv" / "bin" / "python"),
        "-u",
        str(SERVER),
        "--policy-path",
        str(policy),
        "--device",
        device,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            load_seconds = _wait_for_server(process, port, timeout=240)
            with ResourceMonitor(process.pid) as warmup_monitor:
                warmup_fixture = fixtures[0]
                _post(port, "/reset", {"seed": warmup_fixture["seed"]}, 30)
                _, warmup_seconds = _post(port, "/infer", _payload(warmup_fixture), 600)
            with ResourceMonitor(process.pid) as monitor:
                measurements = []
                for fixture in fixtures:
                    _post(port, "/reset", {"seed": fixture["seed"]}, 30)
                    response, latency_seconds = _post(port, "/infer", _payload(fixture), 600)
                    if "error" in response:
                        raise RuntimeError(f"{backend} inference failed: {response['error']}")
                    actions = np.asarray(response["action_chunk"], dtype=np.float32)
                    measurements.append(
                        {
                            "fixture": fixture["name"],
                            "seed": fixture["seed"],
                            "latency_seconds": latency_seconds,
                            "action_shape": list(actions.shape),
                            "action_sha256": _sha256_bytes(actions.tobytes()),
                            "actions": actions.tolist(),
                        }
                    )
            return {
                "backend": backend,
                "server_pid": process.pid,
                "load_seconds": load_seconds,
                "warmup_seconds": warmup_seconds,
                "warmup_peak_gpu_mib": warmup_monitor.peak_gpu_mib,
                "warmup_peak_rss_mib": warmup_monitor.peak_rss_mib,
                "peak_gpu_mib": monitor.peak_gpu_mib,
                "peak_rss_mib": monitor.peak_rss_mib,
                "measurements": measurements,
                "log": str(log_path.relative_to(ROOT)),
            }
        finally:
            _stop_process(process)


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _compare(cpu: dict, hybrid: dict) -> list[dict]:
    rows = []
    for cpu_row, hybrid_row in zip(cpu["measurements"], hybrid["measurements"], strict=True):
        if cpu_row["fixture"] != hybrid_row["fixture"]:
            raise RuntimeError("backend fixture order mismatch")
        reference = np.asarray(cpu_row["actions"], dtype=np.float64).reshape(-1)
        candidate = np.asarray(hybrid_row["actions"], dtype=np.float64).reshape(-1)
        difference = candidate - reference
        denominator = float(np.linalg.norm(reference))
        cosine_denominator = denominator * float(np.linalg.norm(candidate))
        rows.append(
            {
                "fixture": cpu_row["fixture"],
                "max_abs": float(np.max(np.abs(difference))),
                "mean_abs": float(np.mean(np.abs(difference))),
                "rmse": float(np.sqrt(np.mean(np.square(difference)))),
                "relative_l2": float(np.linalg.norm(difference) / max(denominator, 1e-12)),
                "cosine_similarity": float(
                    np.dot(reference, candidate) / max(cosine_denominator, 1e-12)
                ),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--port", type=int, default=18445)
    parser.add_argument("--cpu-threads", type=int, default=24)
    parser.add_argument(
        "--candidate",
        choices=CANDIDATE_BACKENDS,
        default="hybrid_cuda_expert",
    )
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--max-abs-drift", type=float, default=0.05)
    parser.add_argument("--max-mean-abs-drift", type=float, default=0.01)
    parser.add_argument("--min-cosine-similarity", type=float, default=0.995)
    parser.add_argument("--max-candidate-gpu-mib", type=float, default=7500.0)
    args = parser.parse_args()

    policy = args.policy.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fixtures = _fixtures()
    fixture_manifest = [
        {key: value for key, value in fixture.items() if key not in {"image", "state"}}
        for fixture in fixtures
    ]
    runs = {}
    for backend in ("cpu", args.candidate):
        print(f"[pi05-backend-pilot] running {backend}", flush=True)
        runs[backend] = _run_backend(
            backend,
            policy,
            fixtures,
            args.port,
            output.parent,
            args.cpu_threads,
        )

    comparisons = _compare(runs["cpu"], runs[args.candidate])
    cpu_latencies = [row["latency_seconds"] for row in runs["cpu"]["measurements"]]
    candidate_latencies = [
        row["latency_seconds"] for row in runs[args.candidate]["measurements"]
    ]
    speedup = statistics.median(cpu_latencies) / statistics.median(candidate_latencies)
    gates = {
        "speedup": speedup >= args.min_speedup,
        "max_abs_drift": max(row["max_abs"] for row in comparisons) <= args.max_abs_drift,
        "mean_abs_drift": max(row["mean_abs"] for row in comparisons)
        <= args.max_mean_abs_drift,
        "cosine_similarity": min(row["cosine_similarity"] for row in comparisons)
        >= args.min_cosine_similarity,
        "gpu_memory": runs[args.candidate]["peak_gpu_mib"]
        <= args.max_candidate_gpu_mib,
    }
    summary = {
        "cpu_latency_median_seconds": statistics.median(cpu_latencies),
        "cpu_latency_p95_seconds": _percentile(cpu_latencies, 95),
        "candidate_backend": args.candidate,
        "candidate_latency_median_seconds": statistics.median(candidate_latencies),
        "candidate_latency_p95_seconds": _percentile(candidate_latencies, 95),
        "speedup": speedup,
        "max_abs_drift": max(row["max_abs"] for row in comparisons),
        "max_mean_abs_drift": max(row["mean_abs"] for row in comparisons),
        "min_cosine_similarity": min(row["cosine_similarity"] for row in comparisons),
        "qualified": all(gates.values()),
    }
    report = {
        "schema_version": 1,
        "kind": "pi05_backend_qualification_pilot",
        "paper_evidence": False,
        "generated_at_unix": time.time(),
        "source_revision": _git_revision(),
        "implementation_sha256": _implementation_sha256(),
        "policy": str(policy),
        "fixtures": fixture_manifest,
        "thresholds": {
            "min_speedup": args.min_speedup,
            "max_abs_drift": args.max_abs_drift,
            "max_mean_abs_drift": args.max_mean_abs_drift,
            "min_cosine_similarity": args.min_cosine_similarity,
            "max_candidate_gpu_mib": args.max_candidate_gpu_mib,
        },
        "runs": runs,
        "comparisons": comparisons,
        "gates": gates,
        "summary": summary,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"[pi05-backend-pilot] wrote {output}", flush=True)
    return 0 if summary["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
