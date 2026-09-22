#!/usr/bin/env python3
"""Fail-closed audit of the frozen PI0.5/SONIC HumanoidArena baseline.

The six-task primary set is kept separate from diagnostic OpenDoor. This
recompiles all selected raw episodes and videos, then binds the result to the
exact policy, SONIC, simulator, prompt-route, and seed-plan locks. It does not
claim a paired comparison with another architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.compile_humanoidarena_pi05_int8_amendment as compiler  # noqa: E402
from scripts.run_humanoidarena_baseline_matrix_fast import (  # noqa: E402
    MODES,
    TASKS,
    _derive_episode_seed,
)


LOCK_DIR = ROOT / "benchmark/locks/pi05_ha"
SUMMARY = ROOT / "results/benchmark/external/pi05_cuda_int8_v1_amended_summary.json"
ORIGINAL_AUDIT = ROOT / "results/benchmark/external/pi05_cuda_int8_v1_audit.json"
BASELINE_DRIVER = ROOT / "scripts/run_humanoidarena_baseline_matrix_fast.py"
SERVER = ROOT / "scripts/serve_humanoidarena_vla_low_memory.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_lock(name: str) -> dict:
    return json.loads((LOCK_DIR / name).read_text(encoding="utf-8"))


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"missing or hash-mismatched locked input: {path}")


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def audit(runtime_root: Path) -> dict:
    checkpoint = load_lock("checkpoints.json")
    sonic = load_lock("sonic.json")
    protocol = load_lock("protocol.json")
    seeds = load_lock("seed_plan.json")
    primary = protocol["primary_tasks"]
    diagnostic = protocol["diagnostic_tasks"]
    if (
        checkpoint["schema_version"] != sonic["schema_version"]
        or checkpoint["schema_version"] != protocol["schema_version"]
        or protocol["schema_version"] != seeds["schema_version"]
        or set(primary) != set(TASKS) - {"open_door"}
        or diagnostic != ["open_door"]
        or protocol["modes"] != list(MODES)
        or protocol["group_seeds"] != [0, 1, 2]
        or protocol["repeats_per_cell"] != 20
        or protocol["primary_episode_count"] != 1440
        or protocol["diagnostic_episode_count"] != 240
        or protocol["max_steps_by_task"] != {
            task: spec["max_steps"] for task, spec in TASKS.items()
        }
    ):
        raise ValueError("frozen six-task protocol differs from the pinned evaluator")
    if git_head(runtime_root / "_vendor/HumanoidArena") != sonic["humanoidarena_revision"]:
        raise ValueError("HumanoidArena revision changed")
    if git_head(runtime_root / "_vendor/IsaacLab-v2.2.0") != sonic["isaaclab_revision"]:
        raise ValueError("Isaac Lab revision changed")
    for path, expected in (
        (SUMMARY, protocol["amended_summary_sha256"]),
        (ORIGINAL_AUDIT, protocol["original_audit_sha256"]),
        (BASELINE_DRIVER, protocol["baseline_driver_sha256"]),
        (ROOT / "scripts/compile_humanoidarena_pi05_int8_amendment.py", protocol["amendment_compiler_sha256"]),
        (SERVER, protocol["server_script_sha256"]),
    ):
        require_hash(path, expected)
    sonic_root = (
        runtime_root / "_vendor/HumanoidArena/GR00T-WholeBodyControl"
        / "gear_sonic_deploy/policy/release"
    )
    require_hash(sonic_root / "model_encoder.onnx", sonic["encoder_sha256"])
    require_hash(sonic_root / "model_decoder.onnx", sonic["decoder_sha256"])
    models_root = runtime_root / "_artifacts/HumanoidArena/models"
    for task, spec in TASKS.items():
        key = "open_door_diagnostic" if task == "open_door" else task
        directory = models_root / spec["model"]
        weights = directory / checkpoint["weight_file"]
        if weights.stat().st_size != checkpoint["weight_size_bytes"]:
            raise ValueError(f"checkpoint size differs: {weights}")
        require_hash(weights, checkpoint["weights_sha256_by_task"][key])
        small_files = [
            {"name": path.name, "sha256": sha256(path)}
            for path in sorted(directory.iterdir())
            if path.is_file() and path.name != checkpoint["weight_file"]
        ]
        if canonical_sha256(small_files) != checkpoint["small_file_manifest_sha256_by_task"][key]:
            raise ValueError(f"checkpoint metadata differs: {directory}")
    original_root = runtime_root / "_artifacts/HumanoidArena/paper-baselines/pi05-sonic-cuda-int8-v1"
    corrected_root = runtime_root / "_artifacts/HumanoidArena/paper-baselines/pi05-sonic-cuda-int8-ppbox-route-corrected-v1"
    runtime_audit = runtime_root / "results/benchmark/external/pi05_cuda_int8_v1_audit.json"
    require_hash(runtime_audit, protocol["original_audit_sha256"])
    compiler.ROOT = runtime_root
    report = compiler.compile_amendment(original_root, corrected_root, runtime_audit)
    recorded = json.loads(SUMMARY.read_text(encoding="utf-8"))
    report.pop("generated_at_utc", None)
    recorded.pop("generated_at_utc", None)
    if report != recorded:
        raise ValueError("fresh raw-evidence compilation differs from the checked-in summary")
    progress = json.loads((corrected_root / "progress.json").read_text(encoding="utf-8"))
    if (
        progress["model_revision"] != checkpoint["upstream_model_revision"]
        or progress["policy_backend"] != protocol["backend"]
        or progress["policy_device"] != protocol["policy_device"]
    ):
        raise ValueError("policy revision, precision, or device differs")
    expected = [
        {
            "task": task,
            "mode": mode,
            "seed": group_seed,
            "repeat_idx": repeat,
            "episode_seed": _derive_episode_seed(TASKS[task]["task_id"], group_seed, repeat),
        }
        for task in TASKS if task in primary
        for mode in MODES
        for group_seed in protocol["group_seeds"]
        for repeat in range(protocol["repeats_per_cell"])
    ]
    if len(expected) != seeds["identity_count"] or canonical_sha256(expected) != seeds["canonical_json_identity_sha256"]:
        raise ValueError("primary seed plan differs from its canonical lock")
    selected = [
        row for row in report["selected_episode_manifest"] if row["task"] in primary
    ]
    observed = {
        (row["task"], row["mode"], row["seed"], row["repeat_idx"])
        for row in selected
    }
    planned = {
        (row["task"], row["mode"], row["seed"], row["repeat_idx"])
        for row in expected
    }
    if len(selected) != len(planned) or observed != planned:
        raise ValueError("selected primary episode IDs differ from the frozen plan")
    metrics = report["primary_six_tasks_excluding_open_door"]
    if metrics["episodes"] != len(expected):
        raise ValueError("primary metric denominator differs from the frozen plan")
    video_manifest = report["selected_video_manifest"]
    primary_videos = [row for row in video_manifest if "/open_door/" not in row["path"]]
    if len(video_manifest) != 168 or len(primary_videos) != 144:
        raise ValueError("representative video sampling differs from the frozen run")
    return {
        "schema_version": 1,
        "audit_passed": True,
        "claim_boundary": "PI0.5/SONIC HA baseline only; no cross-method paired or CPU/INT8 equivalence claim",
        "primary_tasks": primary,
        "diagnostic_tasks": diagnostic,
        "episodes_observed": len(expected),
        "diagnostic_episodes": protocol["diagnostic_episode_count"],
        "successes": metrics["successes"],
        "success_rate": metrics["success_rate"],
        "success_rate_wilson_95": metrics["success_rate_wilson_95"],
        "failures_by_reason": metrics["result_reason_counts"],
        "primary_video_files_verified": len(primary_videos),
        "diagnostic_video_files_verified": len(video_manifest) - len(primary_videos),
        "amended_summary_sha256": sha256(SUMMARY),
        "original_audit_sha256": sha256(ORIGINAL_AUDIT),
        "checkpoint_lock_sha256": sha256(LOCK_DIR / "checkpoints.json"),
        "sonic_lock_sha256": sha256(LOCK_DIR / "sonic.json"),
        "protocol_lock_sha256": sha256(LOCK_DIR / "protocol.json"),
        "seed_plan_lock_sha256": sha256(LOCK_DIR / "seed_plan.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite release audit: {args.output}")
    result = audit(args.runtime_root.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
