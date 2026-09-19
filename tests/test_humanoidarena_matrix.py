from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_humanoidarena_baseline_matrix.py"
SPEC = importlib.util.spec_from_file_location("humanoidarena_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MATRIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATRIX)


def test_default_matrix_has_1680_baseline_episodes() -> None:
    cells = len(MATRIX.TASKS) * len(MATRIX.MODES) * len(MATRIX.DEFAULT_SEEDS)
    assert cells == 84
    assert cells * 20 == 1680


def test_runtime_env_uses_requested_cpu_capacity() -> None:
    env = MATRIX._runtime_env(cpu_threads=32, compile_threads=32)
    assert env["OMP_NUM_THREADS"] == "32"
    assert env["MKL_NUM_THREADS"] == "32"
    assert env["OPENBLAS_NUM_THREADS"] == "32"
    assert env["NUMEXPR_NUM_THREADS"] == "32"
    assert env["OMP_DYNAMIC"] == "FALSE"
    assert env["MKL_DYNAMIC"] == "FALSE"
    assert env["TORCHINDUCTOR_COMPILE_THREADS"] == "32"
    assert MATRIX.DEFAULT_CPU_THREADS == len(os.sched_getaffinity(0))


def test_cell_complete_requires_all_successful_processes(tmp_path: Path) -> None:
    summary = tmp_path / "summary.jsonl"
    rows = [
        {
            "seed": 1,
            "repeat_idx": repeat_idx,
            "returncode": 0,
            "failure_reason": "timeout",
        }
        for repeat_idx in range(3)
    ]
    summary.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert MATRIX.cell_complete(tmp_path, seed=1, repeats=3)

    rows[-1]["failure_reason"] = "process_error"
    summary.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert not MATRIX.cell_complete(tmp_path, seed=1, repeats=3)
