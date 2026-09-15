"""Locked multi-seed training and validation selection for the HA GR00T bridge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1:
        raise ValueError("HumanoidArena training lock schema_version must be 1")
    if lock.get("training_seeds") != [0, 1, 2]:
        raise ValueError("HumanoidArena training seeds must remain [0, 1, 2]")
    training = lock.get("training", {})
    if training.get("candidate_steps") != [100, 200, 300]:
        raise ValueError("candidate checkpoint steps must remain [100, 200, 300]")
    if training.get("max_steps") != 300 or training.get("save_steps") != 100:
        raise ValueError("training/save step contract changed")
    splits = lock.get("dataset", {}).get("splits", {})
    if [splits.get(name, {}).get("episodes") for name in ("train", "validation", "heldout")] != [
        490,
        70,
        140,
    ]:
        raise ValueError("dataset split contract changed")
    return lock


def verify_training_inputs(repo_root: Path, lock: dict[str, Any]) -> dict[str, Any]:
    base_root = repo_root / lock["base_model"]["path"]
    checked: dict[str, str] = {}
    for relative, expected in lock["base_model"]["signatures"].items():
        path = base_root / relative
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"base model signature mismatch for {relative}: {actual}")
        checked[str(path.relative_to(repo_root))] = actual
    dataset_root = repo_root / lock["dataset"]["path"]
    manifest_path = dataset_root / "split_manifest.json"
    manifest_hash = _sha256(manifest_path)
    if manifest_hash != lock["dataset"]["manifest_sha256"]:
        raise ValueError(f"dataset manifest signature mismatch: {manifest_hash}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("modality_sha256") != lock["dataset"]["modality_sha256"]:
        raise ValueError("dataset modality signature mismatch")
    for split, expected in lock["dataset"]["splits"].items():
        actual = manifest["outputs"][split]
        if any(int(actual[key]) != int(expected[key]) for key in ("episodes", "frames")):
            raise ValueError(f"dataset {split} totals differ from lock")
    return {"base_signatures": checked, "dataset_manifest_sha256": manifest_hash}


def seed_directory(repo_root: Path, lock: dict[str, Any], seed: int) -> Path:
    if seed not in lock["training_seeds"]:
        raise ValueError(f"seed {seed} is not locked")
    return repo_root / lock["output_root"] / f"seed-{seed}"


def training_complete(directory: Path, max_steps: int = 300) -> bool:
    path = directory / "checkpoints" / f"checkpoint-{max_steps}" / "trainer_state.json"
    if not path.is_file():
        return False
    return int(json.loads(path.read_text(encoding="utf-8")).get("global_step", -1)) == max_steps


def build_training_command(
    repo_root: Path, python: Path, lock: dict[str, Any], seed: int
) -> list[str]:
    training = lock["training"]
    output = seed_directory(repo_root, lock, seed) / "checkpoints"
    command = [
        str(python),
        str(repo_root / "scripts/train_gr00t_vla_adapter.py"),
        "--base-model-path",
        str(repo_root / lock["base_model"]["path"]),
        "--dataset-path",
        str(repo_root / lock["dataset"]["path"] / "train"),
        "--modality-config-path",
        str(repo_root / lock["modality_config_path"]),
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
    seed: int,
    step: int,
) -> list[str]:
    if step not in lock["training"]["candidate_steps"]:
        raise ValueError(f"checkpoint step {step} is not locked")
    selection = lock["checkpoint_selection"]
    model = seed_directory(repo_root, lock, seed) / "checkpoints" / f"checkpoint-{step}"
    output = (
        repo_root
        / lock["output_root"]
        / "validation"
        / f"seed-{seed}"
        / f"checkpoint-{step}"
    )
    return [
        str(python),
        str(repo_root / "scripts/evaluate_vla_retraining.py"),
        "--model-path",
        str(model),
        "--dataset-path",
        str(repo_root / lock["dataset"]["path"] / selection["split"]),
        "--output-dir",
        str(output),
        "--trajectory-ids",
        *[str(value) for value in range(70)],
        "--conditions",
        *selection["conditions"],
        "--execution-horizon",
        str(selection["execution_horizon"]),
        "--steps",
        str(selection["frames_per_trajectory"]),
        "--denoising-steps",
        str(selection["denoising_steps"]),
        "--seed",
        str(20260915 + seed * 10_000 + step),
    ]


def build_hidden_command(
    repo_root: Path,
    python: Path,
    lock: dict[str, Any],
    seed: int,
    selected_step: int,
) -> list[str]:
    if selected_step not in lock["training"]["candidate_steps"]:
        raise ValueError(f"checkpoint step {selected_step} is not locked")
    hidden = lock["hidden_evaluation"]
    model = (
        seed_directory(repo_root, lock, seed)
        / "checkpoints"
        / f"checkpoint-{selected_step}"
    )
    output = repo_root / lock["output_root"] / "hidden" / f"seed-{seed}"
    return [
        str(python),
        str(repo_root / "scripts/evaluate_vla_retraining.py"),
        "--model-path",
        str(model),
        "--dataset-path",
        str(repo_root / lock["dataset"]["path"] / "heldout"),
        "--output-dir",
        str(output),
        "--trajectory-ids",
        *[str(value) for value in range(140)],
        "--conditions",
        *hidden["conditions"],
        "--execution-horizon",
        str(hidden["execution_horizon"]),
        "--steps",
        str(hidden["frames_per_trajectory"]),
        "--denoising-steps",
        str(hidden["denoising_steps"]),
        "--seed",
        str(20260915 + seed * 10_000),
    ]


def resource_blockers(lock: dict[str, Any], own_script_name: str = "") -> list[str]:
    """Return reasons exclusive-GPU work must not start."""

    blockers = []
    pattern = lock["resource_exclusion"]["blocking_process_pattern"]
    process = subprocess.run(["pgrep", "-af", pattern], text=True, capture_output=True)
    live = [
        line
        for line in process.stdout.splitlines()
        if not own_script_name or own_script_name not in line
    ]
    if live:
        blockers.append(f"blocking benchmark process is live: {live[0]}")
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    used = sum(int(line.strip()) for line in query.stdout.splitlines() if line.strip())
    limit = int(lock["resource_exclusion"]["maximum_preexisting_compute_vram_mib"])
    if used > limit:
        blockers.append(f"pre-existing compute VRAM is {used} MiB, limit is {limit} MiB")
    return blockers


def evaluation_complete(path: Path, expected_trajectories: int, conditions: list[str]) -> bool:
    metrics = path / "metrics.json"
    if not metrics.is_file():
        return False
    rows = json.loads(metrics.read_text(encoding="utf-8")).get("rows", [])
    identities = {
        (int(row.get("trajectory_id", -1)), str(row.get("condition", ""))) for row in rows
    }
    expected = {
        (trajectory, condition)
        for trajectory in range(expected_trajectories)
        for condition in conditions
    }
    return len(rows) == len(expected) and identities == expected


def validate_selection(report: dict[str, Any], lock: dict[str, Any]) -> int:
    """Verify the immutable validation decision before exposing hidden data."""

    if report.get("selection_split") != "validation" or report.get("hidden_test_accessed") is not False:
        raise ValueError("selection report does not prove validation-only selection")
    selected = report.get("selected_step")
    if type(selected) is not int or selected not in lock["training"]["candidate_steps"]:
        raise ValueError("selection report contains an unlocked checkpoint step")
    if report.get("dataset_manifest_sha256") != lock["dataset"]["manifest_sha256"]:
        raise ValueError("selection report uses a different dataset manifest")
    core = {key: value for key, value in report.items() if key != "selection_sha256"}
    encoded = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != report.get("selection_sha256"):
        raise ValueError("selection report hash mismatch")
    return selected


def select_global_checkpoint(
    repo_root: Path, lock: dict[str, Any], evaluation_root: Path
) -> dict[str, Any]:
    """Select one common step using only macro-per-task validation MSE."""

    metadata_path = repo_root / lock["dataset"]["path"] / "validation/meta/episodes.jsonl"
    metadata = [json.loads(line) for line in metadata_path.read_text().splitlines() if line]
    task_by_episode = {
        int(row["episode_index"]): str(row["source_dataset"]).split("/", 1)[0]
        for row in metadata
    }
    if set(task_by_episode) != set(range(70)):
        raise ValueError("validation metadata must contain dense episode IDs 0..69")

    candidates: dict[str, Any] = {}
    for step in lock["training"]["candidate_steps"]:
        seed_macros = []
        seed_rows = {}
        for seed in lock["training_seeds"]:
            path = evaluation_root / f"seed-{seed}/checkpoint-{step}/metrics.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            rows = report.get("rows", [])
            by_task: dict[str, list[float]] = {}
            identities = set()
            for row in rows:
                identity = (int(row["trajectory_id"]), str(row["condition"]))
                if identity in identities:
                    raise ValueError(f"duplicate validation row in {path}: {identity}")
                identities.add(identity)
                if identity[1] != "clean" or identity[0] not in task_by_episode:
                    raise ValueError(f"unlocked validation row in {path}: {identity}")
                by_task.setdefault(task_by_episode[identity[0]], []).append(float(row["mse"]))
            if len(identities) != 70 or set(by_task) != set(task_by_episode.values()):
                raise ValueError(f"incomplete validation matrix: {path}")
            task_means = {
                task: statistics.fmean(values) for task, values in sorted(by_task.items())
            }
            macro = statistics.fmean(task_means.values())
            seed_macros.append(macro)
            seed_rows[str(seed)] = {"macro_task_mse": macro, "task_mse": task_means}
        candidates[str(step)] = {
            "mean_seed_macro_task_mse": statistics.fmean(seed_macros),
            "seeds": seed_rows,
        }
    selected = min(
        lock["training"]["candidate_steps"],
        key=lambda step: (candidates[str(step)]["mean_seed_macro_task_mse"], step),
    )
    core = {
        "schema_version": 1,
        "selection_split": "validation",
        "hidden_test_accessed": False,
        "criterion": lock["checkpoint_selection"]["criterion"],
        "tie_break": lock["checkpoint_selection"]["tie_break"],
        "selected_step": selected,
        "candidates": candidates,
        "dataset_manifest_sha256": lock["dataset"]["manifest_sha256"],
    }
    encoded = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**core, "selection_sha256": hashlib.sha256(encoded).hexdigest()}
