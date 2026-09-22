"""Fail-closed PI0.5 fine-tuning command for a local LeRobot v3 dataset."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
    batch_size: int = 1
    num_workers: int = 4
    source_dataset: str | None = None


def _selected_episodes(job: PI05TrainingJob, info: dict) -> list[int] | None:
    if "hrvla_packed_video_export" not in info:
        return None
    if info.get("hrvla_subtask_relabel", {}).get("split") != "train":
        raise ValueError("PI0.5 ST-RT can only train on the audited train split")
    audit_path = job.dataset / "meta/hrvla_export_audit.json"
    provenance = job.dataset / "meta/hrvla_source_episodes.jsonl"
    if not audit_path.is_file() or not provenance.is_file():
        raise FileNotFoundError("verified v3 export audit/provenance is missing")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(provenance.read_bytes()).hexdigest()
    if audit.get("status") != "loader_pass" or audit.get("provenance_sha256") != digest:
        raise ValueError("v3 dataset has not passed the LeRobot loader audit")
    if not job.source_dataset or job.source_dataset not in job.base_policy.parts:
        raise ValueError("source dataset and task-matched base checkpoint are required")
    rows = [json.loads(line) for line in provenance.read_text(encoding="utf-8").splitlines()
            if line]
    episodes = [int(row["episode_index"]) for row in rows
                if str(row["source_dataset"]).split("/", 1)[0] == job.source_dataset]
    if not episodes:
        raise ValueError(f"no training episodes for {job.source_dataset}")
    return episodes


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
    if job.output_dir.exists() and any(job.output_dir.iterdir()):
        raise FileExistsError("output directory is not empty; use a fresh run or explicit resume")
    if job.seed < 0 or min(job.steps, job.batch_size, job.num_workers) < 1:
        raise ValueError("seed must be nonnegative and training dimensions positive")
    _selected_episodes(job, info)


def build_training_command(job: PI05TrainingJob) -> list[str]:
    validate_training_job(job)
    info = json.loads((job.dataset / "meta/info.json").read_text(encoding="utf-8"))
    episodes = _selected_episodes(job, info)
    command = [
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
        "--policy.train_expert_only=true",
        "--wandb.enable=false",
        f"--seed={job.seed}",
        f"--batch_size={job.batch_size}",
        f"--num_workers={job.num_workers}",
        f"--steps={job.steps}",
        f"--output_dir={job.output_dir}",
        f"--job_name={job.method_id}-{job.source_dataset or 'all'}-seed-{job.seed}",
    ]
    if episodes is not None:
        command.append(f"--dataset.episodes={json.dumps(episodes, separators=(',', ':'))}")
    return command
