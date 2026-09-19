#!/usr/bin/env python3
"""Profile steady-state PI0.5 CPU latency without changing benchmark episodes."""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from run_humanoidarena_baseline_matrix_fast import (
    POLICY_PYTHON,
    ROOT,
    SERVER,
    TASKS,
    _server_env,
    _terminate_group,
    _wait_for_server,
)


def _payload() -> bytes:
    height, width, channels = 480, 640, 3
    image = bytes(height * width * channels)
    request = {
        "observation": {
            "images": {
                "front": {
                    "shape": [height, width, channels],
                    "dtype": "uint8",
                    "data_b64": base64.b64encode(image).decode("ascii"),
                }
            },
            "state": [0.0] * 64,
        },
        "robot_type": "unitree_g1_refpose_v3_1",
        "task": "HSI_boxing",
        "return_chunk": True,
    }
    return json.dumps(request).encode("utf-8")


def _post(port: int, path: str, payload: bytes, timeout: float) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/{path}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _profile_one(
    *,
    model_path: Path,
    output_dir: Path,
    threads: int,
    interop_threads: int,
    compile_threads: int,
    warmup_requests: int,
    measured_requests: int,
    port: int,
    request_timeout: float,
) -> dict:
    log_path = output_dir / f"threads-{threads}.log"
    command = [
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
    body = _payload()
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
            env=_server_env(threads, interop_threads, compile_threads),
            start_new_session=True,
        )
        try:
            started = time.perf_counter()
            _wait_for_server(port, request_timeout)
            load_seconds = time.perf_counter() - started
            warmup_seconds = []
            measured_seconds = []
            for index in range(warmup_requests + measured_requests):
                _post(port, "reset", json.dumps({"seed": 20260915}).encode(), 10)
                started = time.perf_counter()
                response = _post(port, "infer", body, request_timeout)
                elapsed = time.perf_counter() - started
                if int(response.get("chunk_size", 0)) != 20:
                    raise RuntimeError(f"unexpected action chunk: {response.keys()}")
                target = warmup_seconds if index < warmup_requests else measured_seconds
                target.append(elapsed)
        finally:
            _terminate_group(process, timeout=10)
    return {
        "threads": threads,
        "interop_threads": interop_threads,
        "compile_threads": compile_threads,
        "load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "measured_seconds": measured_seconds,
        "mean_seconds": statistics.fmean(measured_seconds),
        "median_seconds": statistics.median(measured_seconds),
        "requests_per_minute": 60.0 / statistics.fmean(measured_seconds),
        "log_path": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=ROOT / "_artifacts" / "HumanoidArena" / "models",
    )
    parser.add_argument("--threads", nargs="+", type=int, default=[8, 16, 24, 32])
    parser.add_argument("--interop-threads", type=int, default=2)
    parser.add_argument("--compile-threads", type=int, default=32)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--measured-requests", type=int, default=2)
    parser.add_argument("--port", type=int, default=18443)
    parser.add_argument("--request-timeout", type=float, default=1200.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "_artifacts" / "HumanoidArena" / "performance" / "pi05-cpu",
    )
    args = parser.parse_args()
    if any(value < 1 for value in args.threads):
        parser.error("thread counts must be positive")
    if args.warmup_requests < 1 or args.measured_requests < 1:
        parser.error("warmup and measured request counts must be positive")

    model_path = args.model_root.resolve() / TASKS["boxing"]["model"]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for threads in args.threads:
        print(f"[pi05-profile] threads={threads}", flush=True)
        row = _profile_one(
            model_path=model_path,
            output_dir=output_dir,
            threads=threads,
            interop_threads=args.interop_threads,
            compile_threads=args.compile_threads,
            warmup_requests=args.warmup_requests,
            measured_requests=args.measured_requests,
            port=args.port,
            request_timeout=args.request_timeout,
        )
        rows.append(row)
        print(
            f"[pi05-profile] threads={threads} median={row['median_seconds']:.3f}s "
            f"rpm={row['requests_per_minute']:.2f}",
            flush=True,
        )

    best = min(rows, key=lambda row: row["median_seconds"])
    report = {
        "schema_version": 1,
        "scope": "performance-only synthetic input; not benchmark evidence",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "precision": "unchanged released float32 CPU policy",
        "rows": rows,
        "selected_threads": best["threads"],
        "selection_metric": "lowest steady-state median request latency",
        "environment": {
            "logical_cpu_count": os.cpu_count(),
            "interop_threads": args.interop_threads,
            "compile_threads": args.compile_threads,
        },
    }
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[pi05-profile] selected_threads={best['threads']} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
