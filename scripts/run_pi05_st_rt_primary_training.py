#!/usr/bin/env python3
"""Resume-safe task-matched PI0.5 ST-RT training for HA primary tasks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "boxing": ("HSI_boxing", "pi/HSI_boxing/pi05_sonic_boxing_0529/100000/pretrained_model"),
    "doubledesk": ("HOI_double_desk", "pi/HOI_double_desk/pi05_sonic_doubledesk_0529/100000/pretrained_model"),
    "football": ("HOI_football", "pi/HOI_football/pi05_sonic_football_0529/100000/pretrained_model"),
    "pp_box": ("HOI_pp_box", "pi/HOI_pp_box/pi05_sonic_ppbox_0529/100000/pretrained_model"),
    "sit_sofa": ("HSI_sit_sofa", "pi/HSI_sit_sofa/pi05_sonic_sitsofa_0529/100000/pretrained_model"),
    "vision_navi": ("HSI_vision_navi", "pi/HSI_vision_navi/pi05_sonic_visionnavi_0529/100000/pretrained_model"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _checkpoint(output: Path) -> Path:
    return output / "checkpoints/000300/pretrained_model"


def _validate_checkpoint(path: Path) -> dict:
    config = path / "config.json"
    weights = path / "model.safetensors"
    if not config.is_file() or not weights.is_file():
        raise FileNotFoundError(f"incomplete PI0.5 checkpoint: {path}")
    payload = json.loads(config.read_text(encoding="utf-8"))
    if payload.get("type") != "pi05":
        raise ValueError(f"checkpoint is not PI0.5: {path}")
    return {
        "path": str(path),
        "config_sha256": _sha256(config),
        "weights_sha256": _sha256(weights),
        "weights_bytes": weights.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.seed != 0 or args.steps != 300:
        parser.error("the frozen primary training contract is seed 0 and 300 steps")
    if args.batch_size != 8 or args.num_workers != 4:
        parser.error("the profiled RTX 5070 Ti contract is batch 8 with four workers")

    runtime = args.runtime_root.resolve(strict=True)
    artifact_root = args.artifact_root.resolve()
    dataset = runtime / "_artifacts/datasets/humanoidarena-st-rt-v3/train"
    lerobot = runtime / "_vendor/HumanoidArena/lerobot"
    python = lerobot / ".venv/bin/python"
    tokenizer = runtime / "_artifacts/HumanoidArena/tokenizers/paligemma-3b-pt-224"
    model_root = runtime / "_artifacts/HumanoidArena/models"
    for required in (dataset, lerobot, python, tokenizer, model_root):
        if not required.exists():
            raise FileNotFoundError(required)

    outputs = {
        task: (
            artifact_root / "pp_box/pilot-300-seed0-v3-corrected"
            if task == "pp_box"
            else artifact_root / task / "primary-300-seed0-v3-corrected"
        )
        for task in args.tasks
    }
    completed: dict[str, dict] = {}
    progress_path = artifact_root / "primary-training-progress.json"

    def progress(active: str | None = None) -> None:
        _write_json_atomic(progress_path, {
            "schema_version": 1,
            "method_id": "pi05_st_rt",
            "training_seed": 0,
            "steps": 300,
            "batch_size": 8,
            "num_workers": 4,
            "tasks_expected": list(args.tasks),
            "tasks_completed": sorted(completed),
            "active_task": active,
            "complete": len(completed) == len(args.tasks),
            "checkpoints": completed,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    for task in args.tasks:
        output = outputs[task]
        checkpoint = _checkpoint(output)
        if checkpoint.is_dir():
            completed[task] = _validate_checkpoint(checkpoint)
            progress()
            continue
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(
                f"incomplete nonempty output requires manual quarantine: {output}"
            )
        source_dataset, relative_base = TASKS[task]
        command = [
            sys.executable, str(ROOT / "scripts/train_pi05_architecture.py"),
            "--method-id", "pi05_st_rt",
            "--dataset", str(dataset),
            "--base-policy", str(model_root / relative_base),
            "--output-dir", str(output),
            "--lerobot-root", str(lerobot),
            "--python", str(python),
            "--tokenizer-dir", str(tokenizer),
            "--source-dataset", source_dataset,
            "--seed", "0", "--steps", "300", "--batch-size", "8",
            "--num-workers", "4", "--log-freq", "20",
        ]
        progress(task)
        print(json.dumps({"task": task, "command": command}), flush=True)
        if args.dry_run:
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        log_path = output.with_suffix(".log")
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            result = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"{task} training failed with {result.returncode}: {log_path}")
        completed[task] = _validate_checkpoint(checkpoint)
        progress()
    progress()
    print(json.dumps(json.loads(progress_path.read_text(encoding="utf-8")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
