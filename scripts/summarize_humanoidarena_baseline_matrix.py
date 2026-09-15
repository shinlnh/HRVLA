#!/usr/bin/env python3
"""Validate and summarize the released HumanoidArena baseline matrix."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "_artifacts" / "HumanoidArena" / "paper-baselines" / "pi05-sonic"
DEFAULT_OUTPUT = ROOT / "results" / "humanoidarena" / "baseline-matrix" / "summary.json"
TASKS = ("boxing", "doubledesk", "football", "open_door", "pp_box", "sit_sofa", "vision_navi")
MODES = ("base_test", "semantic", "vision", "execution")
SEEDS = (0, 1, 2)
REPEATS = 20
SOURCE_REVISION = "68479287a784a69be9ce6ad739311d2f11f75ef9"
MODEL_REVISION = "da13e072902840e2682afde360b763f1edb76d32"
DATASET_REVISION = "a079beddd6b1521f762c991be8f36993f17ebeca"
ISAACLAB_REVISION = "46dff135f44683f031edf346e544fcfd8456b2bb"


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        return [0.0, 0.0]
    probability = successes / trials
    denominator = 1 + z**2 / trials
    center = (probability + z**2 / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(probability * (1 - probability) / trials + z**2 / (4 * trials**2))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def collect_rows(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    invalid_cells: list[str] = []
    for task in TASKS:
        for mode in MODES:
            for seed in SEEDS:
                cell_name = f"{mode}/{task}/seed-{seed}"
                cell_rows = _read_jsonl(root / cell_name / "summary.jsonl")
                repeat_ids = {int(row.get("repeat_idx", -1)) for row in cell_rows}
                valid = (
                    len(cell_rows) == REPEATS
                    and repeat_ids == set(range(REPEATS))
                    and all(
                        int(row.get("seed", -1)) == seed
                        and int(row.get("returncode", -1)) == 0
                        and row.get("failure_reason") != "process_error"
                        for row in cell_rows
                    )
                )
                if not valid:
                    invalid_cells.append(cell_name)
                    continue
                for row in cell_rows:
                    row = dict(row)
                    row.update({"task_key": task, "mode": mode})
                    rows.append(row)
    return rows, invalid_cells


def _statistics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    successes = sum(bool(row.get("success")) for row in materialized)
    trials = len(materialized)
    successful_steps = [
        int(row["episode_steps"])
        for row in materialized
        if bool(row.get("success")) and row.get("episode_steps") is not None
    ]
    reasons = Counter(
        str(row.get("failure_reason") or ("success" if row.get("success") else "unknown"))
        for row in materialized
    )
    return {
        "episodes": trials,
        "successes": successes,
        "failures": trials - successes,
        "success_rate": successes / trials if trials else 0.0,
        "success_rate_wilson_95": wilson_interval(successes, trials),
        "mean_success_steps": (
            sum(successful_steps) / len(successful_steps) if successful_steps else None
        ),
        "result_reason_counts": dict(sorted(reasons.items())),
    }


def summarize(root: Path, *, allow_partial: bool = False) -> dict[str, Any]:
    rows, invalid_cells = collect_rows(root)
    if invalid_cells and not allow_partial:
        raise ValueError(
            f"baseline matrix is incomplete: {len(invalid_cells)}/84 invalid cells; "
            f"first={invalid_cells[0]}"
        )

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, str, int, int]] = set()
    duplicate_identities: list[list[Any]] = []
    for row in rows:
        task = str(row["task_key"])
        mode = str(row["mode"])
        seed = int(row["seed"])
        repeat = int(row["repeat_idx"])
        identity = (task, mode, seed, repeat)
        if identity in identities:
            duplicate_identities.append(list(identity))
        identities.add(identity)
        by_task[task].append(row)
        by_mode[mode].append(row)
        by_task_mode[f"{mode}/{task}"].append(row)
        by_seed[str(seed)].append(row)

    expected_episodes = len(TASKS) * len(MODES) * len(SEEDS) * REPEATS
    complete = not invalid_cells and len(rows) == expected_episodes and not duplicate_identities
    return {
        "schema_version": 1,
        "benchmark": "HumanoidArena",
        "method": "released PI0.5 task checkpoint with SONIC whole-body controller",
        "includes_ours": False,
        "claim_status": (
            "matrix_complete_admission_pending" if complete else "partial_non_claim"
        ),
        "admission_blockers": [
            "locked upstream OpenDoor self-test contract mismatch",
            "missing independent 20/20 oracle admission evidence",
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": SOURCE_REVISION,
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "isaaclab_revision": ISAACLAB_REVISION,
        "matrix": {
            "tasks": list(TASKS),
            "modes": list(MODES),
            "seeds": list(SEEDS),
            "repeats_per_seed": REPEATS,
            "cells_expected": 84,
            "cells_complete": 84 - len(invalid_cells),
            "episodes_expected": expected_episodes,
            "episodes_complete": len(rows),
            "invalid_cells": invalid_cells,
            "duplicate_identities": duplicate_identities,
        },
        "overall": _statistics(rows),
        "by_task": {key: _statistics(value) for key, value in sorted(by_task.items())},
        "by_mode": {key: _statistics(value) for key, value in sorted(by_mode.items())},
        "by_task_mode": {key: _statistics(value) for key, value in sorted(by_task_mode.items())},
        "by_seed": {key: _statistics(value) for key, value in sorted(by_seed.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    report = summarize(args.input_root.resolve(), allow_partial=args.allow_partial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[summary] status={report['claim_status']} "
        f"episodes={report['matrix']['episodes_complete']}/"
        f"{report['matrix']['episodes_expected']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
