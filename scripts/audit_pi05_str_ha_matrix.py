#!/usr/bin/env python3
"""Audit every paired PI0.5-STR HA episode, method trace, and sampled video."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_humanoidarena_baseline_matrix_fast import (  # noqa: E402
    MODES, TASKS, _derive_episode_seed,
)

LOCKS = ROOT / "benchmark/locks/pi05_ha"
INVALID_INFRASTRUCTURE = {"interrupted", "process_error", "sim_error"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _inside(root: Path, value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"evidence path escapes matrix root: {path}")
    return path


def _head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * ((p * (1 - p) + z * z / (4 * total)) / total) ** 0.5 / denominator
    return [centre - radius, centre + radius]


def audit(runtime_root: Path, matrix_root: Path, *, include_diagnostic: bool = False) -> dict:
    protocol = json.loads((LOCKS / "protocol.json").read_text(encoding="utf-8"))
    seed_lock = json.loads((LOCKS / "seed_plan.json").read_text(encoding="utf-8"))
    sonic = json.loads((LOCKS / "sonic.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((LOCKS / "checkpoints.json").read_text(encoding="utf-8"))
    contract = json.loads((ROOT / "benchmark/v2_contract.json").read_text(encoding="utf-8"))
    tasks = protocol["primary_tasks"]
    if include_diagnostic:
        tasks = protocol["diagnostic_tasks"]
    if (
        contract["architecture_token"] != "PI05-STR"
        or contract["suite"] != "HA"
        or contract["precision"] != "cuda_int8"
        or protocol["backend"] != "cuda_int8_weight_only"
        or protocol["modes"] != list(MODES)
        or protocol["group_seeds"] != [0, 1, 2]
        or protocol["repeats_per_cell"] != 20
        or set(tasks) != (set(TASKS) - {"open_door"} if not include_diagnostic else {"open_door"})
    ):
        raise ValueError("PI0.5-STR matrix contract differs from frozen baseline")
    if _head(runtime_root / "_vendor/HumanoidArena") != sonic["humanoidarena_revision"]:
        raise ValueError("HumanoidArena source revision differs")
    if _head(runtime_root / "_vendor/IsaacLab-v2.2.0") != sonic["isaaclab_revision"]:
        raise ValueError("Isaac Lab source revision differs")
    if sha256(runtime_root / "scripts/run_humanoidarena_baseline_matrix_fast.py") != protocol["baseline_driver_sha256"]:
        raise ValueError("baseline orchestration driver differs")
    sonic_root = (
        runtime_root / "_vendor/HumanoidArena/GR00T-WholeBodyControl"
        / "gear_sonic_deploy/policy/release"
    )
    for name, expected_hash in (
        ("model_encoder.onnx", sonic["encoder_sha256"]),
        ("model_decoder.onnx", sonic["decoder_sha256"]),
    ):
        if sha256(sonic_root / name) != expected_hash:
            raise ValueError(f"SONIC {name} differs from frozen baseline")
    models_root = runtime_root / "_artifacts/HumanoidArena/models"
    for task in tasks:
        key = "open_door_diagnostic" if task == "open_door" else task
        model_dir = models_root / TASKS[task]["model"]
        weights = model_dir / checkpoint["weight_file"]
        if (
            weights.stat().st_size != checkpoint["weight_size_bytes"]
            or sha256(weights) != checkpoint["weights_sha256_by_task"][key]
        ):
            raise ValueError(f"PI0.5 checkpoint differs: {task}")
        small_files = [
            {"name": path.name, "sha256": sha256(path)}
            for path in sorted(model_dir.iterdir())
            if path.is_file() and path.name != checkpoint["weight_file"]
        ]
        if canonical_sha256(small_files) != checkpoint["small_file_manifest_sha256_by_task"][key]:
            raise ValueError(f"PI0.5 checkpoint metadata differs: {task}")
    expected = [
        {
            "task": task, "mode": mode, "seed": group_seed,
            "repeat_idx": repeat,
            "episode_seed": _derive_episode_seed(TASKS[task]["task_id"], group_seed, repeat),
        }
        for task in TASKS if task in tasks
        for mode in MODES
        for group_seed in protocol["group_seeds"]
        for repeat in range(protocol["repeats_per_cell"])
    ]
    if not include_diagnostic and canonical_sha256(expected) != seed_lock["canonical_json_identity_sha256"]:
        raise ValueError("paired primary episode plan differs from frozen seed lock")
    progress = json.loads((matrix_root / "progress.json").read_text(encoding="utf-8"))
    if (
        progress.get("method_id") != "pi05_str"
        or progress.get("policy_backend") != protocol["backend"]
        or progress.get("policy_device") != "cuda:0"
        or progress.get("implementation_revision") != contract["execution_revision"]
    ):
        raise ValueError("matrix progress has wrong method/backend/revision")
    results: list[dict] = []
    attempts: list[dict] = []
    program_hashes: set[str] = set()
    videos = 0
    transitions = 0
    for identity in expected:
        task, mode, seed, repeat = (
            identity["task"], identity["mode"], identity["seed"], identity["repeat_idx"]
        )
        cell = matrix_root / mode / task / f"seed-{seed}"
        matching = []
        for path in sorted((cell / "episodes").glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("repeat_idx") == repeat:
                matching.append((path, row))
        valid = [
            (path, row) for path, row in matching
            if row.get("failure_reason") not in INVALID_INFRASTRUCTURE
            and int(row.get("returncode", 0)) == 0
        ]
        if len(valid) != 1:
            raise ValueError(f"expected exactly one valid atomic row for {identity}; got {len(valid)}")
        for path, row in matching:
            if (path, row) not in valid:
                attempts.append({"identity": identity, "path": str(path), "sha256": sha256(path)})
        path, row = valid[0]
        if (
            row.get("seed") != seed
            or row.get("episode_seed") != identity["episode_seed"]
            or row.get("task") != TASKS[task]["task_id"]
            or row.get("max_steps") != TASKS[task]["max_steps"]
            or type(row.get("success")) is not bool
            or not 0 < int(row.get("episode_steps", 0)) <= TASKS[task]["max_steps"]
            or row.get("model_path") != str(models_root / TASKS[task]["model"])
        ):
            raise ValueError(f"episode metadata differs from plan: {identity}")
        method = row.get("hrvla_method")
        if not isinstance(method, dict) or (
            method.get("method_id") != "pi05_str"
            or method.get("implementation_revision") != contract["execution_revision"]
            or method.get("policy_requests", 0) < 1
            or method.get("policy_reset_seed") != identity["episode_seed"]
        ):
            raise ValueError(f"STR method summary missing or mismatched: {identity}")
        trace_path = _inside(matrix_root, method["trace_path"])
        trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
        decisions = [item for item in trace if item.get("event") == "instruction_selected"]
        if (
            len(decisions) != method["policy_requests"]
            or any(item.get("policy_action_dim") != 40 for item in decisions)
            or canonical_sha256(trace) != method["trace_sha256"]
            or json.loads((trace_path.parent / "method-summary.json").read_text(encoding="utf-8")) != method
        ):
            raise ValueError(f"STR trace differs from episode result: {identity}")
        program_hashes.add(method["program_sha256"])
        transitions += int(method["completed_transition_count"])
        video_hash = None
        if row.get("video_recorded"):
            video_path = _inside(matrix_root, row["video_path"])
            if not video_path.is_file() or video_path.stat().st_size == 0:
                raise ValueError(f"sampled video is missing: {identity}")
            video_hash = sha256(video_path)
            videos += 1
        results.append({
            **identity,
            "success": row["success"],
            "failure_reason": row.get("failure_reason"),
            "episode_steps": row["episode_steps"],
            "completed_transition_count": method["completed_transition_count"],
            "policy_requests": method["policy_requests"],
            "episode_path": str(path), "episode_sha256": sha256(path),
            "trace_path": str(trace_path), "trace_sha256": sha256(trace_path),
            "video_path": row.get("video_path") if video_hash else None,
            "video_sha256": video_hash,
        })
    frozen_program = json.loads(
        (ROOT / "benchmark/humanoidarena_method_programs.json").read_text(encoding="utf-8")
    )
    if program_hashes != {canonical_sha256(frozen_program)}:
        raise ValueError("STR program hash differs from checked-in method program")
    expected_count = 240 if include_diagnostic else 1440
    if len(results) != expected_count:
        raise ValueError("episode denominator differs from frozen protocol")
    by_task = {}
    for task in tasks:
        task_rows = [row for row in results if row["task"] == task]
        successes = sum(row["success"] for row in task_rows)
        by_task[task] = {
            "episodes": len(task_rows), "successes": successes,
            "success_rate": successes / len(task_rows),
            "wilson_95": _wilson(successes, len(task_rows)),
            "failure_reasons": dict(Counter(str(row["failure_reason"]) for row in task_rows)),
        }
    successes = sum(row["success"] for row in results)
    return {
        "schema_version": 1,
        "audit_passed": True,
        "method_id": "pi05_str",
        "claim_boundary": "PI0.5-STR HA paired nominal result only; recovery-injection outcomes remain a separate protocol",
        "diagnostic": include_diagnostic,
        "architecture_revision": contract["architecture_revision"],
        "implementation_revision": contract["execution_revision"],
        "program_sha256": next(iter(program_hashes)),
        "episodes_observed": len(results),
        "successes": successes,
        "success_rate": successes / len(results),
        "success_rate_wilson_95": _wilson(successes, len(results)),
        "completed_subtask_transitions": transitions,
        "videos_verified": videos,
        "excluded_infrastructure_attempts": attempts,
        "by_task": by_task,
        "episode_manifest": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.output}")
    result = audit(
        args.runtime_root.resolve(strict=True), args.matrix_root.resolve(strict=True),
        include_diagnostic=args.diagnostic,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "episode_manifest"}, indent=2))


if __name__ == "__main__":
    main()
