"""Build and run the locked multi-seed VLA retraining matrix."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def load_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1:
        raise ValueError("paper retraining lock schema_version must be 1")
    seeds = lock.get("training_seeds")
    if seeds != [0, 1, 2]:
        raise ValueError("paper retraining seeds must remain [0, 1, 2]")
    if set(lock.get("methods", {})) != {"ST-RT", "STR-RT"}:
        raise ValueError("paper retraining lock must define ST-RT and STR-RT")
    return lock


def run_directory(repo_root: Path, lock: dict[str, Any], method: str, seed: int) -> Path:
    output_name = lock["methods"][method]["output_name"]
    return repo_root / lock["output_root"] / f"{output_name}-seed-{seed}"


def is_complete(directory: Path, max_steps: int) -> bool:
    state_path = directory / "checkpoints" / f"checkpoint-{max_steps}" / "trainer_state.json"
    if not state_path.is_file():
        return False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return int(state.get("global_step", -1)) == max_steps


def build_command(
    repo_root: Path,
    python: Path,
    lock: dict[str, Any],
    method: str,
    seed: int,
) -> list[str]:
    if method not in lock["methods"]:
        raise ValueError(f"unknown method: {method}")
    if seed not in lock["training_seeds"]:
        raise ValueError(f"seed {seed} is not locked")
    shared = lock["shared"]
    config = lock["methods"][method]
    output = run_directory(repo_root, lock, method, seed) / "checkpoints"
    command = [
        str(python),
        str(repo_root / "scripts/train_gr00t_vla_adapter.py"),
        "--base-model-path",
        str(repo_root / lock["base_model_path"]),
        "--dataset-path",
        str(repo_root / config["dataset_path"]),
        "--modality-config-path",
        str(repo_root / lock["modality_config_path"]),
        "--output-dir",
        str(output),
        "--max-steps",
        str(shared["max_steps"]),
        "--save-steps",
        str(shared["save_steps"]),
        "--global-batch-size",
        str(config["global_batch_size"]),
        "--gradient-accumulation-steps",
        str(config["gradient_accumulation_steps"]),
        "--dataloader-num-workers",
        str(config["dataloader_num_workers"]),
        "--learning-rate",
        str(shared["learning_rate"]),
        "--seed",
        str(seed),
        "--state-dropout-prob",
        str(config["state_dropout_probability"]),
    ]
    if shared["color_jitter"]:
        command.append("--color-jitter")
    return command


def run_matrix(
    repo_root: Path,
    python: Path,
    lock: dict[str, Any],
    methods: list[str],
    seeds: list[int],
    *,
    dry_run: bool = False,
) -> int:
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    max_steps = int(lock["shared"]["max_steps"])
    for method in methods:
        for seed in seeds:
            directory = run_directory(repo_root, lock, method, seed)
            if is_complete(directory, max_steps):
                print(f"skip complete: {method} seed={seed}")
                continue
            command = build_command(repo_root, python, lock, method, seed)
            print(f"run: {method} seed={seed}", flush=True)
            if dry_run:
                print(" ".join(command))
                continue
            directory.mkdir(parents=True, exist_ok=True)
            log_path = directory / "train.log"
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=repo_root,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                    log.flush()
                return_code = process.wait()
            if return_code:
                return return_code
            if not is_complete(directory, max_steps):
                print(f"error: completion marker missing for {method} seed={seed}", file=sys.stderr)
                return 2
    return 0
