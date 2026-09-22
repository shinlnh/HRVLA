#!/usr/bin/env python3
"""Resume-safe PI0.5-ST HumanoidArena matrix on the frozen PI0.5 episode plan.

The baseline driver owns seeding, horizons, CUDA INT8 server lifetime, video
sampling, atomic results, and resume. Only the evaluator entrypoint changes:
the ST wrapper installs a per-episode observed-state language planner hook.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/run_humanoidarena_pi05_st_episode.py"


def _load_baseline(runtime_root: Path):
    path = runtime_root / "scripts/run_humanoidarena_baseline_matrix_fast.py"
    spec = importlib.util.spec_from_file_location("hrvla_pi05_st_baseline_driver", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen baseline driver: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _implementation_revision() -> str:
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--short", "--untracked-files=no"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("PI0.5 ST implementation must be committed before benchmark")
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    runtime_root = args.runtime_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    baseline = _load_baseline(runtime_root)
    revision = _implementation_revision()
    original_sim_command = baseline._sim_command
    original_progress = baseline._matrix_progress

    def st_sim_command(*, task_name, mode, batch_path, port,
                       record_video_every_n, step_log_every_n):
        command = original_sim_command(
            task_name=task_name, mode=mode, batch_path=batch_path, port=port,
            record_video_every_n=record_video_every_n,
            step_log_every_n=step_log_every_n,
        )
        method_output = (
            output_root / "method-traces" / task_name / mode
            / f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{time.time_ns()}"
        )
        command[2] = str(WRAPPER)
        command[3:3] = [
            "--runtime-root", str(runtime_root),
            "--method-output-dir", str(method_output),
        ]
        return command

    def st_progress(*args, **kwargs):
        progress = original_progress(*args, **kwargs)
        progress["method_id"] = "pi05_st"
        progress["implementation_revision"] = revision
        return progress

    baseline._sim_command = st_sim_command
    baseline._matrix_progress = st_progress
    sys.argv = [sys.argv[0], "--output-root", str(output_root), *forwarded]
    print(json.dumps({
        "method_id": "pi05_st",
        "implementation_revision": revision,
        "runtime_root": str(runtime_root),
        "output_root": str(output_root),
        "baseline_driver": str(runtime_root / "scripts/run_humanoidarena_baseline_matrix_fast.py"),
        "forwarded": forwarded,
    }, sort_keys=True), flush=True)
    return int(baseline.main())


if __name__ == "__main__":
    raise SystemExit(main())
