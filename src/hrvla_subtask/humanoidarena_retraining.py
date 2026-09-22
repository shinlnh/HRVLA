"""Guarded 40-D HumanoidArena retraining contract for GR00T architecture rows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class GR00TRetrainJob:
    method_id: str
    python: Path
    train_script: Path
    dataset: Path
    base_checkpoint: Path
    modality_config: Path
    output_dir: Path
    seed: int
    steps: int = 300
    workers: int = 16
    batch_size: int = 4
    accumulation: int = 16


def validate_job(job: GR00TRetrainJob) -> None:
    if job.method_id not in {"gr00t_st_rt", "gr00t_str_rt"}:
        raise ValueError("unknown GR00T retraining method")
    for path in (job.python, job.train_script, job.modality_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (job.base_checkpoint / "config.json").is_file():
        raise FileNotFoundError("shared HumanoidArena GR00T checkpoint is missing")
    if not (job.base_checkpoint / "model.safetensors.index.json").is_file():
        raise FileNotFoundError("shared HumanoidArena GR00T weights are missing")
    info_path = job.dataset / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    for key, shape in (("action", [40]), ("observation.state", [64])):
        if features.get(key, {}).get("shape") != shape:
            raise ValueError(f"{key} is not the HumanoidArena {shape} contract")
    tasks_path = job.dataset / "meta/tasks.jsonl"
    if not tasks_path.is_file() or not tasks_path.read_text(encoding="utf-8").strip():
        raise FileNotFoundError("subtask-labelled tasks.jsonl is missing")
    if not (job.dataset / "data").is_dir():
        raise FileNotFoundError("LeRobot training data is missing")
    if job.method_id == "gr00t_str_rt":
        labels_path = job.dataset / "meta/hrvla_recovery_labels.json"
        if not labels_path.is_file():
            raise FileNotFoundError("STR-RT requires audited recovery-label metadata")
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        if labels.get("schema_version") != 1 or labels.get("recovery_samples", 0) < 1:
            raise ValueError("STR-RT has no valid recovery-labelled samples")
    if job.output_dir.exists() and any(job.output_dir.iterdir()):
        raise FileExistsError("output directory is not empty; use a fresh run")
    if job.seed < 0 or min(job.steps, job.workers, job.batch_size, job.accumulation) < 1:
        raise ValueError("seed must be nonnegative and training dimensions positive")


def build_command(job: GR00TRetrainJob) -> list[str]:
    validate_job(job)
    return [
        str(job.python), str(job.train_script),
        "--base-model-path", str(job.base_checkpoint),
        "--dataset-path", str(job.dataset),
        "--modality-config-path", str(job.modality_config),
        "--output-dir", str(job.output_dir),
        "--max-steps", str(job.steps),
        "--save-steps", "100",
        "--global-batch-size", str(job.batch_size),
        "--gradient-accumulation-steps", str(job.accumulation),
        "--dataloader-num-workers", str(job.workers),
        "--seed", str(job.seed),
    ]
