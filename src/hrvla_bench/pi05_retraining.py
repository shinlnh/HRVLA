"""Fail-closed PI0.5 fine-tuning command for a local LeRobot v3 dataset."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class PI05TrainingJob:
    method_id: str
    dataset: Path
    base_policy: Path
    output_dir: Path
    train_script: Path
    python: Path
    seed: int = 0
    steps: int = 100_000
    batch_size: int = 8
    num_workers: int = 8


def validate_training_job(job: PI05TrainingJob) -> None:
    if job.method_id not in {"pi05_st_rt", "pi05_str_rt"}:
        raise ValueError(f"unsupported PI0.5 retraining method: {job.method_id}")
    if not job.python.is_file() or not job.train_script.is_file():
        raise FileNotFoundError("LeRobot Python or training entry point is missing")
    if not job.base_policy.is_dir():
        raise FileNotFoundError("local PI0.5 base checkpoint is missing")
    if not (job.base_policy / "config.json").is_file():
        raise FileNotFoundError("PI0.5 base checkpoint config.json is missing")
    config = json.loads((job.base_policy / "config.json").read_text(encoding="utf-8"))
    if config.get("type") != "pi05":
        raise ValueError("base checkpoint is not a PI0.5 policy")
    info_path = job.dataset / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError("LeRobot dataset meta/info.json is missing")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError("PI0.5 trainer requires LeRobot v3.0; convert a copy of the v2.1 dataset")
    features = info.get("features", {})
    for key, expected in (("action", [40]), ("observation.state", [64])):
        if features.get(key, {}).get("shape") != expected:
            raise ValueError(f"{key} must have shape {expected}")
    if not (job.dataset / "meta" / "tasks.parquet").is_file():
        raise FileNotFoundError("LeRobot v3.0 tasks.parquet is missing")
    if not (job.dataset / "data").is_dir():
        raise FileNotFoundError("LeRobot dataset data directory is missing")
    if job.method_id == "pi05_str_rt":
        labels_path = job.dataset / "meta" / "hrvla_recovery_labels.json"
        if not labels_path.is_file():
            raise FileNotFoundError("STR-RT requires audited recovery-label metadata")
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        if labels.get("schema_version") != 1 or labels.get("recovery_samples", 0) < 1:
            raise ValueError("STR-RT recovery-label metadata has no valid recovery samples")
        if not labels.get("scenario_ids"):
            raise ValueError("STR-RT recovery-label metadata lacks scenario IDs")
    if job.output_dir.exists() and any(job.output_dir.iterdir()):
        raise FileExistsError("output directory is not empty; use a fresh run or explicit resume")
    if job.seed < 0 or min(job.steps, job.batch_size, job.num_workers) < 1:
        raise ValueError("seed must be nonnegative and training dimensions positive")


def build_training_command(job: PI05TrainingJob) -> list[str]:
    validate_training_job(job)
    return [
        str(job.python),
        str(job.train_script),
        f"--dataset.repo_id=local/{job.dataset.name}",
        f"--dataset.root={job.dataset}",
        "--dataset.image_transforms.enable=false",
        "--policy.type=pi05",
        f"--policy.pretrained_path={job.base_policy}",
        "--policy.device=cuda",
        "--policy.max_state_dim=64",
        "--policy.max_action_dim=40",
        "--policy.n_obs_steps=1",
        "--policy.chunk_size=20",
        "--policy.n_action_steps=20",
        "--policy.optimizer_lr=2.5e-5",
        "--policy.push_to_hub=false",
        "--policy.gradient_checkpointing=true",
        "--policy.dtype=bfloat16",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=false",
        "--wandb.enable=false",
        f"--seed={job.seed}",
        f"--batch_size={job.batch_size}",
        f"--num_workers={job.num_workers}",
        f"--steps={job.steps}",
        f"--output_dir={job.output_dir}",
        f"--job_name={job.method_id}-seed-{job.seed}",
    ]
