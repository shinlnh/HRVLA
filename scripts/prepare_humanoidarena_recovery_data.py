#!/usr/bin/env python3
"""Build STR-RT LeRobot views from admitted, successful oracle demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.plan import canonical_sha256, load_json  # noqa: E402


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def recovery_split(scenario_id: str, trial_indices: list[int]) -> dict[str, list[int]]:
    if sorted(trial_indices) != list(range(20)):
        raise ValueError(f"{scenario_id}: recovery dataset requires trial indices 0..19")
    ranked = sorted(
        trial_indices,
        key=lambda index: hashlib.sha256(
            f"hrvla-recovery-data-v1|{scenario_id}|{index}".encode()
        ).digest(),
    )
    return {
        "train": sorted(ranked[:14]),
        "validation": sorted(ranked[14:17]),
        "heldout": sorted(ranked[17:]),
    }


def _valid_demo_record(record_path: Path) -> dict[str, Any]:
    record = load_json(record_path)
    trial_dir = record_path.parent
    manifest_path = trial_dir / "recovery-demonstration.json"
    arrays_path = trial_dir / "recovery-demonstration.npz"
    manifest = load_json(manifest_path)
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if canonical_sha256(core) != manifest.get("manifest_sha256"):
        raise ValueError(f"demonstration manifest hash differs: {manifest_path}")
    if manifest["manifest_sha256"] != record.get(
        "recovery_demonstration_manifest_sha256"
    ):
        raise ValueError("oracle record and demonstration manifest differ")
    if _file_sha256(arrays_path) != manifest.get("arrays_sha256"):
        raise ValueError(f"demonstration arrays hash differs: {arrays_path}")
    video_path = Path(manifest["video_path"])
    if not video_path.is_file() or _file_sha256(video_path) != manifest.get("video_sha256"):
        raise ValueError(f"demonstration video hash differs: {video_path}")
    if record.get("success") is not True or record.get("failure_reason") != "success":
        raise ValueError("behavioral oracle failure cannot enter recovery training data")
    if manifest.get("eligible_for_recovery_training") is not True:
        raise ValueError("oracle demonstration is not training eligible")
    return {
        "record": record,
        "manifest": manifest,
        "arrays_path": arrays_path,
        "video_path": video_path.resolve(),
        "trial_dir": trial_dir,
    }


def _discover_scenario(
    oracle_root: Path, scenario_id: str, expected_report: dict[str, Any]
) -> list[dict[str, Any]]:
    if expected_report.get("status") != "admitted" or expected_report.get("successes") != 20:
        raise ValueError(f"{scenario_id}: oracle report is not admitted 20/20")
    by_index: dict[int, dict[str, Any]] = {}
    attempts = sorted((oracle_root / scenario_id).glob("attempt-*"), reverse=True)
    for attempt in attempts:
        for record_path in sorted(attempt.glob("trial-*/oracle-record.json")):
            try:
                item = _valid_demo_record(record_path)
            except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
                continue
            index = int(item["record"].get("trial_index", -1))
            if index not in by_index:
                by_index[index] = item
    if set(by_index) != set(range(20)):
        raise ValueError(f"{scenario_id}: valid recovery demonstrations are incomplete")
    expected_hashes = expected_report["episode_result_sha256"]
    for index, item in by_index.items():
        if item["record"]["episode_result_sha256"] != expected_hashes[index]:
            raise ValueError(f"{scenario_id}/trial-{index}: report selects different evidence")
    return [by_index[index] for index in range(20)]


def _stats(frames: list[pd.DataFrame]) -> dict[str, Any]:
    output = {}
    for key in ("observation.state", "action"):
        values = np.vstack(
            [np.stack(frame[key].map(lambda value: np.asarray(value, dtype=np.float32))) for frame in frames]
        )
        output[key] = {
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
    return output


def materialize_recovery_split(
    *,
    split: str,
    suite: dict[str, Any],
    oracle_root: Path,
    output_root: Path,
    template_dataset: Path,
    lock: dict[str, Any],
) -> dict[str, Any]:
    import pandas as pd

    if split not in {"train", "validation"}:
        raise ValueError("heldout recovery demonstrations stay locked until selection")
    target = output_root / split
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing recovery split: {target}")
    scenarios = [
        (task, scenario)
        for task in suite["tasks"]
        for scenario in task["scenarios"]
    ]
    required = int(lock["recovery_rt_dataset"]["required_scenarios"])
    if len(scenarios) != required:
        raise ValueError("recovery suite scenario count differs from the RT lock")
    assignments = {}
    selected = []
    task_rows = []
    for task_index, (task, scenario) in enumerate(scenarios):
        scenario_id = scenario["id"]
        report = load_json(oracle_root / scenario_id / "oracle-report.json")
        demonstrations = _discover_scenario(oracle_root, scenario_id, report)
        assignment = recovery_split(scenario_id, list(range(20)))
        assignments[scenario_id] = assignment
        instruction = demonstrations[0]["manifest"]["instruction"]
        if any(item["manifest"]["instruction"] != instruction for item in demonstrations):
            raise ValueError(f"{scenario_id}: recovery instructions differ across trials")
        task_rows.append({"task_index": task_index, "task": instruction})
        for trial_index in assignment[split]:
            selected.append((task_index, task, scenario, demonstrations[trial_index]))

    building = output_root / f".{split}.building"
    if building.exists():
        raise FileExistsError(f"stale recovery build directory requires audit: {building}")
    (building / "meta").mkdir(parents=True)
    episode_rows = []
    frames = []
    global_index = 0
    for episode_index, (task_index, task, scenario, item) in enumerate(selected):
        arrays = np.load(item["arrays_path"])
        states = np.asarray(arrays["observation_state"], dtype=np.float32)
        actions = np.asarray(arrays["action"], dtype=np.float32)
        video_indices = np.asarray(arrays["video_frame_index"], dtype=np.int64)
        if states.ndim != 2 or states.shape[1] != 64 or actions.shape != (len(states), 40):
            raise ValueError("recovery arrays violate state64/action40")
        if len(states) < 40 or len(video_indices) != len(states):
            raise ValueError("recovery demonstration is too short or video alignment differs")
        if np.any(np.diff(video_indices) != 1):
            raise ValueError("recovery demonstration video frames are not contiguous")
        length = len(states)
        frame = pd.DataFrame(
            {
                "observation.state": list(states),
                "action": list(actions),
                "timestamp": np.arange(length, dtype=np.float32) / 50.0,
                "frame_index": np.arange(length, dtype=np.int64),
                "episode_index": np.full(length, episode_index, dtype=np.int64),
                "index": np.arange(global_index, global_index + length, dtype=np.int64),
                "task_index": np.full(length, task_index, dtype=np.int64),
            }
        )
        global_index += length
        parquet = building / f"data/chunk-000/episode_{episode_index:06d}.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(parquet, index=False)
        frames.append(frame)
        video = (
            building
            / "videos/chunk-000/observation.images.front"
            / f"episode_{episode_index:06d}.mp4"
        )
        video.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(item["video_path"], video)
        manifest = item["manifest"]
        episode_rows.append(
            {
                "episode_index": episode_index,
                "length": length,
                "tasks": [manifest["instruction"]],
                "video_frame_offset": int(video_indices[0]),
                "source_oracle_id": item["record"]["oracle_id"],
                "source_scenario_id": scenario["id"],
                "source_task_id": task["id"],
                "source_trial_index": int(item["record"]["trial_index"]),
                "source_rollout_seed": int(item["record"]["rollout_seed"]),
                "source_demonstration_manifest_sha256": manifest["manifest_sha256"],
                "source_video": str(item["video_path"]),
            }
        )

    template_info = load_json(template_dataset / "train/meta/info.json")
    template_info.update(
        total_episodes=len(episode_rows),
        total_frames=sum(row["length"] for row in episode_rows),
        total_tasks=len(task_rows),
        total_videos=len(episode_rows),
        total_chunks=1,
        fps=50,
    )
    template_info["hrvla_recovery_relabel"] = {
        "schema_version": 1,
        "split": split,
        "training_only": True,
        "after_failure_only": True,
    }
    _write_json(building / "meta/info.json", template_info)
    _write_json(building / "meta/stats.json", _stats(frames))
    _write_json(
        building / "meta/modality.json",
        load_json(template_dataset / "train/meta/modality.json"),
    )
    _write_jsonl(building / "meta/tasks.jsonl", task_rows)
    _write_jsonl(building / "meta/episodes.jsonl", episode_rows)
    os.replace(building, target)
    core = {
        "schema_version": 1,
        "claim_boundary": "successful oracle behavior relabelled for STR-RT training only",
        "split": split,
        "episodes": len(episode_rows),
        "frames": sum(row["length"] for row in episode_rows),
        "scenarios": len(scenarios),
        "assignment_derivation": lock["recovery_rt_dataset"]["split_derivation"],
        "assignments": assignments,
        "demonstration_manifest_sha256": [
            row["source_demonstration_manifest_sha256"] for row in episode_rows
        ],
    }
    manifest = {**core, "manifest_sha256": canonical_sha256(core)}
    manifest_path = output_root / "manifests" / f"{split}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to replace recovery manifest: {manifest_path}")
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "config/humanoidarena-rt-training.lock.json",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=ROOT / "benchmark/suites/hrvla_recovery_v0.json",
    )
    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=ROOT / "_artifacts/HumanoidArena/recovery-admission/oracle",
    )
    parser.add_argument("--split", action="append", choices=("train", "validation"))
    args = parser.parse_args()
    lock = load_json(args.lock.resolve())
    suite = load_json(args.suite.resolve())
    output_root = (ROOT / lock["recovery_rt_dataset"]["path"]).resolve()
    template = (ROOT / lock["source_dataset"]["path"]).resolve()
    if _file_sha256(template / "split_manifest.json") != lock["source_dataset"][
        "manifest_sha256"
    ]:
        raise ValueError("template dataset manifest differs from the RT training lock")
    splits = args.split or ["train", "validation"]
    reports = [
        materialize_recovery_split(
            split=split,
            suite=suite,
            oracle_root=args.oracle_root.resolve(),
            output_root=output_root,
            template_dataset=template,
            lock=lock,
        )
        for split in splits
    ]
    print(
        json.dumps(
            {
                report["split"]: {
                    "episodes": report["episodes"],
                    "frames": report["frames"],
                    "manifest_sha256": report["manifest_sha256"],
                }
                for report in reports
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
