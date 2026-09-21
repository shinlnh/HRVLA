#!/usr/bin/env python3
"""Audit the raw INT8 matrix without promoting its misrouted PickPlaceBox row.

This report is intentionally descriptive.  The old CPU matrix was paused after
356 episodes and is useful as a matched sensitivity check, not a prospective
CPU-vs-INT8 equivalence test or a substitute for the complete GPU matrix.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_humanoidarena_baseline_matrix_fast import (
    MODES,
    TASKS,
    _derive_episode_seed,
)


DEFAULT_GPU = ROOT / "_artifacts/HumanoidArena/paper-baselines/pi05-sonic-cuda-int8-v1"
DEFAULT_CPU = ROOT / "_artifacts/HumanoidArena/paper-baselines/pi05-sonic"
DEFAULT_OUTPUT = ROOT / "results/benchmark/external/pi05_cuda_int8_v1_audit.json"
EXPECTED_TASK_ROUTES = {
    "boxing": "HSI_boxing",
    "doubledesk": "HOI_double_desk",
    "football": "HOI_football",
    "open_door": "HSI_open_door",
    "pp_box": "HOI_pp_box",
    "sit_sofa": "HSI_sit_sofa",
    "vision_navi": "HSI_vision_navi",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_rows(root: Path) -> tuple[dict[tuple[str, str, int, int], dict], list[str], list[dict]]:
    rows = {}
    problems: list[str] = []
    evidence: list[dict] = []
    for task, spec in TASKS.items():
        for mode in MODES:
            for seed in (0, 1, 2):
                cell = root / mode / task / f"seed-{seed}"
                paths = sorted((cell / "episodes").glob("*.json"))
                if len(paths) != 20:
                    problems.append(f"{mode}/{task}/seed-{seed}: expected 20 JSONs, got {len(paths)}")
                runtime_path = cell / "runtime.json"
                if not runtime_path.is_file():
                    problems.append(f"{mode}/{task}/seed-{seed}: missing runtime.json")
                else:
                    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
                    if root == DEFAULT_GPU or root.name == DEFAULT_GPU.name:
                        if runtime.get("policy_backend") != "cuda_int8_weight_only" or runtime.get("policy_device") != "cuda:0":
                            problems.append(f"{mode}/{task}/seed-{seed}: backend/device drift")
                for path in paths:
                    try:
                        row = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        problems.append(f"{path}: unreadable: {exc}")
                        continue
                    repeat = int(row.get("repeat_idx", -1))
                    key = task, mode, seed, repeat
                    if key in rows:
                        problems.append(f"{path}: duplicate identity {key}")
                    rows[key] = row
                    if repeat not in range(20) or int(row.get("seed", -1)) != seed:
                        problems.append(f"{path}: invalid seed/repeat")
                    expected_seed = _derive_episode_seed(str(spec["task_id"]), seed, repeat)
                    if int(row.get("episode_seed", -1)) != expected_seed:
                        problems.append(f"{path}: episode seed drift")
                    if row.get("task") != spec["task_id"] or int(row.get("max_steps", -1)) != int(spec["max_steps"]):
                        problems.append(f"{path}: task/horizon drift")
                    if row.get("failure_reason") in {"interrupted", "sim_error", "process_error"}:
                        problems.append(f"{path}: infrastructure failure counted as outcome")
                    if bool(row.get("success")) != (row.get("failure_reason") == "success"):
                        problems.append(f"{path}: success/reason inconsistency")
                    video = str(row.get("video_path") or "")
                    if bool(row.get("video_recorded")) != bool(video):
                        problems.append(f"{path}: video flag/path mismatch")
                    if video:
                        video_path = Path(video)
                        if not video_path.is_file() or video_path.stat().st_size < 1000:
                            problems.append(f"{path}: missing/tiny video {video}")
                        else:
                            evidence.append({"path": str(video_path.relative_to(ROOT)), "sha256": _sha256(video_path)})
                    evidence.append({"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)})
                if {k[3] for k in rows if k[:3] == (task, mode, seed)} != set(range(20)):
                    problems.append(f"{mode}/{task}/seed-{seed}: incomplete repeat set")
    return rows, problems, evidence


def _task_route_audit(root: Path) -> dict[str, dict]:
    result = {}
    for task in TASKS:
        path = root / "driver-logs" / task / "server.log"
        matches = re.findall(r"first_infer .*?task_name=([^ ]+) task=([^\n]+)", path.read_text(encoding="utf-8") if path.is_file() else "")
        observed = sorted({match[0] for match in matches})
        result[task] = {
            "expected": EXPECTED_TASK_ROUTES[task],
            "observed": observed,
            "passes": bool(observed) and observed == [EXPECTED_TASK_ROUTES[task]],
        }
    return result


def _paired_sensitivity(cpu: dict, gpu: dict) -> dict:
    by_task = defaultdict(list)
    for key in sorted(cpu.keys() & gpu.keys()):
        c, g = cpu[key], gpu[key]
        if (c.get("episode_seed"), c.get("episode_object_seed"), c.get("task"), c.get("max_steps"), c.get("model_path")) != (
            g.get("episode_seed"), g.get("episode_object_seed"), g.get("task"), g.get("max_steps"), g.get("model_path")
        ):
            raise ValueError(f"CPU/GPU pairing mismatch: {key}")
        by_task[key[0]].append((bool(c.get("success")), bool(g.get("success"))))

    def counts(pairs: list[tuple[bool, bool]]) -> dict:
        n = len(pairs)
        cpu_only = sum(c and not g for c, g in pairs)
        gpu_only = sum(g and not c for c, g in pairs)
        return {
            "pairs": n,
            "cpu_successes": sum(c for c, _ in pairs),
            "int8_successes": sum(g for _, g in pairs),
            "cpu_only": cpu_only,
            "int8_only": gpu_only,
            "paired_risk_difference_int8_minus_cpu": (gpu_only - cpu_only) / n if n else None,
        }

    all_pairs = [pair for pairs in by_task.values() for pair in pairs]
    return {
        "claim_boundary": "retrospective matched sensitivity only; not a pre-registered equivalence test",
        "overall": counts(all_pairs),
        "by_task": {task: counts(pairs) for task, pairs in sorted(by_task.items())},
    }


def audit(gpu_root: Path, cpu_root: Path) -> dict:
    progress = json.loads((gpu_root / "progress.json").read_text(encoding="utf-8"))
    gpu, problems, evidence = _atomic_rows(gpu_root)
    if progress.get("complete") is not True or progress.get("episodes_observed") != 1680 or progress.get("policy_backend") != "cuda_int8_weight_only":
        problems.append("GPU progress marker is incomplete or backend-mismatched")
    routes = _task_route_audit(gpu_root)
    cpu = {}
    # The CPU run is deliberately partial. Read only its finished atomic rows.
    for path in cpu_root.glob("*/*/seed-*/episodes/*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        rel = path.relative_to(cpu_root).parts
        cpu[(rel[1], rel[0], int(row["seed"]), int(row["repeat_idx"]))] = row
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "cuda_int8_weight_only",
        "progress": progress,
        "structural_gate": {"passed": not problems, "problems": problems, "atomic_episodes": len(gpu)},
        "prompt_route_gate": {"passed": all(x["passes"] for x in routes.values()), "by_task": routes},
        "raw_outcome_counts": dict(sorted(Counter(str(row.get("failure_reason")) for row in gpu.values()).items())),
        "paired_cpu_sensitivity": _paired_sensitivity(cpu, gpu),
        "evidence_manifest": sorted(evidence, key=lambda x: x["path"]),
        "claim_status": "external_matrix_not_admitted_while_prompt_route_fails" if not all(x["passes"] for x in routes.values()) else "matrix_structurally_complete_admission_pending",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-root", type=Path, default=DEFAULT_GPU)
    parser.add_argument("--cpu-root", type=Path, default=DEFAULT_CPU)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(args.gpu_root.resolve(), args.cpu_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[pi05-audit] structural={report['structural_gate']['passed']} prompt={report['prompt_route_gate']['passed']} episodes={report['structural_gate']['atomic_episodes']} output={args.output}")
    return 0 if report["structural_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
