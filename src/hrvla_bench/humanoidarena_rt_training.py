"""Fail-closed command and selection contracts for HA ST-RT/STR-RT training."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import subprocess
from typing import Any

from .humanoidarena_training import validate_selection as validate_common_selection
from .plan import canonical_sha256


RT_METHODS = ("gr00t_st_rt", "gr00t_str_rt")


def load_rt_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1:
        raise ValueError("RT training lock schema_version must be 1")
    if lock.get("training_seeds") != [0, 1, 2]:
        raise ValueError("RT training seeds must remain [0, 1, 2]")
    if tuple(lock.get("methods", {})) != RT_METHODS:
        raise ValueError("RT training lock must contain ST-RT then STR-RT")
    training = lock.get("training", {})
    if training.get("candidate_steps") != [100, 200, 300]:
        raise ValueError("RT candidate checkpoints must remain 100/200/300")
    if training.get("max_steps") != 300 or training.get("save_steps") != 100:
        raise ValueError("RT train/save step contract differs")
    return lock


def _manifest(lock: dict[str, Any], repo_root: Path, method_id: str, split: str) -> dict[str, Any]:
    family = lock["methods"][method_id]["dataset_family"]
    config = lock[family]
    path = repo_root / config["path"] / "manifests" / f"{split}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = config.get("manifests", {}).get(split)
    if not isinstance(expected, str) or manifest.get("manifest_sha256") != expected:
        raise ValueError(f"{method_id}/{split}: dataset manifest is not frozen")
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if canonical_sha256(core) != expected:
        raise ValueError(f"{method_id}/{split}: dataset manifest hash differs")
    return manifest


def validate_rt_inputs(
    repo_root: Path,
    lock: dict[str, Any],
    common_lock: dict[str, Any],
    common_selection: dict[str, Any],
    method_ids: tuple[str, ...] = RT_METHODS,
) -> int:
    selected_step = validate_common_selection(common_selection, common_lock)
    if lock.get("method_program_sha256") is None:
        raise ValueError("RT method program hash is missing")
    if not method_ids or len(set(method_ids)) != len(method_ids) or any(
        method_id not in RT_METHODS for method_id in method_ids
    ):
        raise ValueError("RT methods must be a non-empty unique locked subset")
    for method_id in method_ids:
        train = _manifest(lock, repo_root, method_id, "train")
        validation = _manifest(lock, repo_root, method_id, "validation")
        expected = int(lock["methods"][method_id]["validation_episodes"])
        if validation.get("episodes") != expected:
            raise ValueError(f"{method_id}: validation episode count differs")
        if int(train.get("episodes", 0)) < expected:
            raise ValueError(f"{method_id}: training dataset is unexpectedly small")
    for seed in lock["training_seeds"]:
        path = (
            repo_root
            / common_lock["output_root"]
            / f"seed-{seed}/checkpoints/checkpoint-{selected_step}"
        )
        if not path.is_dir():
            raise FileNotFoundError(path)
    return selected_step


def run_directory(repo_root: Path, lock: dict[str, Any], method_id: str, seed: int) -> Path:
    if method_id not in RT_METHODS or seed not in lock["training_seeds"]:
        raise ValueError("unlocked RT method or training seed")
    return (
        repo_root
        / lock["output_root"]
        / lock["methods"][method_id]["output_name"]
        / f"seed-{seed}"
    )


def training_complete(path: Path, step: int = 300) -> bool:
    state = path / f"checkpoints/checkpoint-{step}/trainer_state.json"
    if not state.is_file():
        return False
    return int(json.loads(state.read_text(encoding="utf-8")).get("global_step", -1)) == step


def build_training_command(
    repo_root: Path,
    python: Path,
    lock: dict[str, Any],
    common_lock: dict[str, Any],
    common_step: int,
    method_id: str,
    seed: int,
) -> list[str]:
    config = lock["methods"][method_id]
    family = lock[config["dataset_family"]]
    training = lock["training"]
    base = (
        repo_root
        / common_lock["output_root"]
        / f"seed-{seed}/checkpoints/checkpoint-{common_step}"
    )
    output = run_directory(repo_root, lock, method_id, seed) / "checkpoints"
    command = [
        str(python),
        str(repo_root / "scripts/train_gr00t_vla_adapter.py"),
        "--base-model-path",
        str(base),
        "--dataset-path",
        str(repo_root / family["path"] / "train"),
        "--modality-config-path",
        str(repo_root / common_lock["modality_config_path"]),
        "--output-dir",
        str(output),
        "--max-steps",
        str(training["max_steps"]),
        "--save-steps",
        str(training["save_steps"]),
        "--learning-rate",
        str(training["learning_rate"]),
        "--global-batch-size",
        str(training["physical_batch_size"]),
        "--gradient-accumulation-steps",
        str(training["gradient_accumulation_steps"]),
        "--dataloader-num-workers",
        str(training["dataloader_workers"]),
        "--seed",
        str(seed),
        "--state-dropout-prob",
        str(training["state_dropout_probability"]),
    ]
    if training["color_jitter"]:
        command.append("--color-jitter")
    return command


def build_validation_command(
    repo_root: Path,
    python: Path,
    lock: dict[str, Any],
    method_id: str,
    seed: int,
    step: int,
) -> list[str]:
    if step not in lock["training"]["candidate_steps"]:
        raise ValueError("unlocked RT checkpoint step")
    config = lock["methods"][method_id]
    family = lock[config["dataset_family"]]
    count = int(config["validation_episodes"])
    model = run_directory(repo_root, lock, method_id, seed) / f"checkpoints/checkpoint-{step}"
    output = run_directory(repo_root, lock, method_id, seed) / f"validation/checkpoint-{step}"
    return [
        str(python),
        str(repo_root / "scripts/evaluate_vla_retraining.py"),
        "--model-path",
        str(model),
        "--dataset-path",
        str(repo_root / family["path"] / "validation"),
        "--output-dir",
        str(output),
        "--trajectory-ids",
        *[str(index) for index in range(count)],
        "--conditions",
        "clean",
        "--execution-horizon",
        "40",
        "--steps",
        "120",
        "--denoising-steps",
        "4",
        "--seed",
        str(20260915 + seed * 10_000 + step),
    ]


def evaluation_complete(path: Path, expected: int) -> bool:
    metrics = path / "metrics.json"
    if not metrics.is_file():
        return False
    rows = json.loads(metrics.read_text(encoding="utf-8")).get("rows", [])
    identities = {(row.get("trajectory_id"), row.get("condition")) for row in rows}
    return len(rows) == expected and identities == {(index, "clean") for index in range(expected)}


def select_checkpoint(
    repo_root: Path, lock: dict[str, Any], method_id: str
) -> dict[str, Any]:
    config = lock["methods"][method_id]
    family = lock[config["dataset_family"]]
    metadata_path = repo_root / family["path"] / "validation/meta/episodes.jsonl"
    metadata = [json.loads(line) for line in metadata_path.read_text().splitlines() if line]
    task_by_episode = {}
    for row in metadata:
        task = row.get("source_task_id") or str(row.get("source_dataset", "")).split("/", 1)[0]
        if not task:
            raise ValueError("RT validation episode lacks its high-level task identity")
        task_by_episode[int(row["episode_index"])] = task
    expected = int(config["validation_episodes"])
    if set(task_by_episode) != set(range(expected)):
        raise ValueError("RT validation episode IDs are not dense and complete")
    candidates = {}
    for step in lock["training"]["candidate_steps"]:
        seed_macros = []
        seed_reports = {}
        for seed in lock["training_seeds"]:
            path = run_directory(repo_root, lock, method_id, seed) / f"validation/checkpoint-{step}/metrics.json"
            rows = json.loads(path.read_text(encoding="utf-8")).get("rows", [])
            if len(rows) != expected:
                raise ValueError(f"incomplete RT validation metrics: {path}")
            grouped: dict[str, list[float]] = {}
            identities = set()
            for row in rows:
                identity = (int(row["trajectory_id"]), str(row["condition"]))
                if identity in identities or identity[1] != "clean" or identity[0] not in task_by_episode:
                    raise ValueError(f"unlocked or duplicate RT validation row: {identity}")
                identities.add(identity)
                grouped.setdefault(task_by_episode[identity[0]], []).append(float(row["mse"]))
            task_mse = {task: statistics.fmean(values) for task, values in sorted(grouped.items())}
            macro = statistics.fmean(task_mse.values())
            seed_macros.append(macro)
            seed_reports[str(seed)] = {"macro_task_mse": macro, "task_mse": task_mse}
        candidates[str(step)] = {
            "mean_seed_macro_task_mse": statistics.fmean(seed_macros),
            "seeds": seed_reports,
        }
    selected = min(
        lock["training"]["candidate_steps"],
        key=lambda step: (candidates[str(step)]["mean_seed_macro_task_mse"], step),
    )
    manifest = _manifest(lock, repo_root, method_id, "validation")
    core = {
        "schema_version": 1,
        "method_id": method_id,
        "selection_split": "validation",
        "hidden_test_accessed": False,
        "criterion": lock["selection"]["criterion"],
        "tie_break": lock["selection"]["tie_break"],
        "selected_step": selected,
        "candidates": candidates,
        "validation_manifest_sha256": manifest["manifest_sha256"],
    }
    return {**core, "selection_sha256": canonical_sha256(core)}


def resource_blockers(lock: dict[str, Any], own_name: str) -> list[str]:
    blockers = []
    process = subprocess.run(
        ["pgrep", "-af", lock["resource_exclusion"]["blocking_process_pattern"]],
        capture_output=True,
        text=True,
    )
    live = [line for line in process.stdout.splitlines() if own_name not in line]
    if live:
        blockers.append(f"external matrix is active: {live[0]}")
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    used = sum(int(line) for line in query.stdout.splitlines() if line.strip())
    limit = int(lock["resource_exclusion"]["maximum_preexisting_compute_vram_mib"])
    if used > limit:
        blockers.append(f"pre-existing compute VRAM is {used} MiB, limit is {limit} MiB")
    return blockers


__all__ = [
    "RT_METHODS",
    "build_training_command",
    "build_validation_command",
    "evaluation_complete",
    "load_rt_lock",
    "resource_blockers",
    "run_directory",
    "select_checkpoint",
    "training_complete",
    "validate_rt_inputs",
]
