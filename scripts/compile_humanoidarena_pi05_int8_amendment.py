#!/usr/bin/env python3
"""Compile a provenance-explicit PI0.5 INT8 matrix after the PPBox prompt fix.

The original 240 wrong-prompt PickPlaceBox outcomes are never overwritten or
silently pooled. Six unaffected tasks come from the complete original run;
PickPlaceBox comes from a separately labeled exact-protocol rerun.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_humanoidarena_baseline_matrix_fast import MODES, TASKS, _derive_episode_seed
from scripts.summarize_humanoidarena_baseline_matrix import _statistics


ORIGINAL_ROOT = ROOT / "_artifacts/HumanoidArena/paper-baselines/pi05-sonic-cuda-int8-v1"
CORRECTED_ROOT = ROOT / "_artifacts/HumanoidArena/paper-baselines/pi05-sonic-cuda-int8-ppbox-route-corrected-v1"
ORIGINAL_AUDIT = ROOT / "results/benchmark/external/pi05_cuda_int8_v1_audit.json"
DEFAULT_OUTPUT = ROOT / "results/benchmark/external/pi05_cuda_int8_v1_amended_summary.json"
CORRECTED_TASK = "pp_box"
CORRECTED_ROUTE = "HOI_pp_box"
CORRECTED_INSTRUCTION = "Move the box from the table onto the shelf."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_root(task: str, original_root: Path, corrected_root: Path) -> Path:
    return corrected_root if task == CORRECTED_TASK else original_root


def _assert_corrected_route(log_text: str) -> int:
    routes = re.findall(r"first_infer .*?task_name=([^ ]+) task='([^']+)'", log_text)
    if not routes or any(route != (CORRECTED_ROUTE, CORRECTED_INSTRUCTION) for route in routes):
        raise ValueError(f"corrected PickPlaceBox server prompt is not invariant: {set(routes)}")
    return len(routes)


def _validate_launch(launch: dict) -> None:
    if launch.get("kind") != "corrected_ppbox_language_route_rerun" or launch.get("expected_episodes") != 240:
        raise ValueError("corrected-task launch manifest is absent or malformed")
    revision = str(launch.get("implementation_revision", ""))
    if not revision:
        raise ValueError("corrected-task implementation revision is missing")
    for relative_path, expected_digest in launch.get("script_sha256", {}).items():
        source = subprocess.run(
            ["git", "show", f"{revision}:{relative_path}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(source).hexdigest() != expected_digest:
            raise ValueError(f"launch manifest script hash mismatch: {relative_path}")


def compile_amendment(
    original_root: Path,
    corrected_root: Path,
    original_audit_path: Path,
) -> dict:
    original_audit = json.loads(original_audit_path.read_text(encoding="utf-8"))
    if not original_audit.get("structural_gate", {}).get("passed"):
        raise ValueError("original 1,680-episode structural audit did not pass")
    routes = original_audit["prompt_route_gate"]["by_task"]
    if routes[CORRECTED_TASK]["passes"] or any(
        not row["passes"] for task, row in routes.items() if task != CORRECTED_TASK
    ):
        raise ValueError("the prompt-route failure is not isolated to PickPlaceBox")
    recorded_hashes = {
        row["path"]: row["sha256"] for row in original_audit["evidence_manifest"]
    }
    corrected_progress = json.loads((corrected_root / "progress.json").read_text(encoding="utf-8"))
    original_progress = original_audit["progress"]
    if (
        corrected_progress.get("complete") is not True
        or corrected_progress.get("episodes_observed") != 240
        or corrected_progress.get("episodes_expected") != 240
        or corrected_progress.get("cells_completed") != 12
        or corrected_progress.get("policy_backend") != "cuda_int8_weight_only"
        or any(
            corrected_progress.get(field) != original_progress.get(field)
            for field in ("model_revision", "source_revision", "isaaclab_revision", "policy_device")
        )
    ):
        raise ValueError("corrected task did not finish the same locked INT8 protocol")
    launch_path = corrected_root / "launch.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    _validate_launch(launch)
    log_path = corrected_root / "driver-logs" / CORRECTED_TASK / "server.log"
    prompt_requests = _assert_corrected_route(log_path.read_text(encoding="utf-8"))

    selected: list[dict] = []
    source_manifest: list[dict] = []
    video_manifest: list[dict] = []
    by_task = defaultdict(list)
    by_task_mode = defaultdict(list)
    by_mode = defaultdict(list)
    by_seed = defaultdict(list)
    for task, spec in TASKS.items():
        source_root = _selected_root(task, original_root, corrected_root)
        for mode in MODES:
            for seed in (0, 1, 2):
                cell = source_root / mode / task / f"seed-{seed}"
                files = sorted((cell / "episodes").glob("*.json"))
                if len(files) != 20:
                    raise ValueError(f"{cell}: expected 20 selected episodes, got {len(files)}")
                repeats = set()
                for path in files:
                    row = json.loads(path.read_text(encoding="utf-8"))
                    repeat = int(row.get("repeat_idx", -1))
                    if repeat not in range(20) or repeat in repeats:
                        raise ValueError(f"{path}: repeat ID missing or duplicated")
                    repeats.add(repeat)
                    if (
                        int(row.get("seed", -1)) != seed
                        or int(row.get("episode_seed", -1)) != _derive_episode_seed(str(spec["task_id"]), seed, repeat)
                        or row.get("task") != spec["task_id"]
                        or int(row.get("max_steps", -1)) != int(spec["max_steps"])
                        or row.get("failure_reason") in {"interrupted", "sim_error", "process_error"}
                    ):
                        raise ValueError(f"{path}: selected episode violates locked protocol")
                    relative_path = str(path.relative_to(ROOT))
                    digest = _sha256(path)
                    if task != CORRECTED_TASK and recorded_hashes.get(relative_path) != digest:
                        raise ValueError(f"{path}: unaffected original evidence changed after audit")
                    video = str(row.get("video_path") or "")
                    if bool(row.get("video_recorded")) != bool(video):
                        raise ValueError(f"{path}: video flag/path mismatch")
                    if video:
                        video_path = Path(video)
                        if not video_path.is_file() or video_path.stat().st_size < 1000:
                            raise ValueError(f"{path}: missing or tiny video {video}")
                        video_relative = str(video_path.relative_to(ROOT))
                        video_digest = _sha256(video_path)
                        if task != CORRECTED_TASK and recorded_hashes.get(video_relative) != video_digest:
                            raise ValueError(f"{video_path}: unaffected original video changed after audit")
                        video_manifest.append({"path": video_relative, "sha256": video_digest})
                    source_manifest.append({
                        "task": task,
                        "mode": mode,
                        "seed": seed,
                        "repeat_idx": repeat,
                        "source": "corrected_prompt_rerun" if task == CORRECTED_TASK else "original_int8_matrix",
                        "path": relative_path,
                        "sha256": digest,
                    })
                    selected.append(row)
                    by_task[task].append(row)
                    by_task_mode[f"{mode}/{task}"].append(row)
                    by_mode[mode].append(row)
                    by_seed[str(seed)].append(row)
                if repeats != set(range(20)):
                    raise ValueError(f"{cell}: incomplete selected repeat set")

    if len(selected) != 1680:
        raise ValueError(f"expected exactly 1,680 selected rows, got {len(selected)}")
    if len(video_manifest) != 168:
        raise ValueError(f"expected exactly 168 selected videos, got {len(video_manifest)}")
    primary = [row for task, rows in by_task.items() if task != "open_door" for row in rows]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "released task-specific PI0.5 + SONIC, CUDA INT8 weight-only",
        "claim_status": "amended_int8_matrix_complete_cpu_equivalence_not_proven",
        "amendment": {
            "reason": "original PickPlaceBox Gym ID was passed as the language prompt",
            "retained_invalid_route_root": str(original_root.relative_to(ROOT)),
            "original_audit_path": str(original_audit_path.relative_to(ROOT)),
            "original_audit_sha256": _sha256(original_audit_path),
            "corrected_root": str(corrected_root.relative_to(ROOT)),
            "corrected_launch_sha256": _sha256(launch_path),
            "corrected_implementation_revision": launch["implementation_revision"],
            "corrected_prompt_requests_observed": prompt_requests,
            "affected_task": CORRECTED_TASK,
            "affected_original_episodes_retained": 240,
            "selected_corrected_episodes": 240,
        },
        "matrix": {"cells": 84, "episodes": 1680, "tasks": list(TASKS), "modes": list(MODES), "seeds": [0, 1, 2], "repeats": 20},
        "overall": _statistics(selected),
        "primary_six_tasks_excluding_open_door": _statistics(primary),
        "by_task": {task: _statistics(rows) for task, rows in sorted(by_task.items())},
        "by_task_mode": {key: _statistics(rows) for key, rows in sorted(by_task_mode.items())},
        "by_mode": {mode: _statistics(rows) for mode, rows in sorted(by_mode.items())},
        "by_seed": {seed: _statistics(rows) for seed, rows in sorted(by_seed.items())},
        "selected_episode_manifest": sorted(source_manifest, key=lambda row: (row["task"], row["mode"], row["seed"], row["repeat_idx"])),
        "selected_video_manifest": sorted(video_manifest, key=lambda row: row["path"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, default=ORIGINAL_ROOT)
    parser.add_argument("--corrected-root", type=Path, default=CORRECTED_ROOT)
    parser.add_argument("--original-audit", type=Path, default=ORIGINAL_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = compile_amendment(args.original_root.resolve(), args.corrected_root.resolve(), args.original_audit.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[pi05-amendment] status={report['claim_status']} episodes={report['matrix']['episodes']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
