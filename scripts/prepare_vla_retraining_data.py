#!/usr/bin/env python3
"""Create leakage-resistant train/held-out Arena G1 subtask datasets.

NVIDIA's public snapshot contains sparse episode IDs while the current GR00T
loader indexes episodes densely.  This tool also fixes that incompatibility by
materializing deterministic dense IDs.  Subtask labels are derived from the
left-hand close/release events and the 40-step action horizon, so each language
instruction describes the outcome represented by its target action chunk.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
from typing import Iterable

import numpy as np
import pandas as pd


SUBTASKS = (
    "Move the left hand toward the apple while maintaining a stable stance.",
    "Reach to and securely grasp the apple with the left hand.",
    "Lift and carry the apple toward the plate while keeping the left hand closed.",
    "Move the apple over the plate, release it, and retract the left hand safely.",
)

RECOVERY_SUBTASKS = (
    "Recover from a missed grasp by re-approaching the apple with the left hand.",
    "Recover the grasp by aligning and securely closing the left hand around the apple.",
    "Stabilize the recovered apple and carry it toward the plate.",
    "Complete recovery by placing the apple on the plate and retracting safely.",
)


def split_episode_ids(
    ids: Iterable[int], heldout_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Return deterministic disjoint train and held-out IDs."""
    ordered = sorted(set(ids))
    if not ordered:
        raise ValueError("no episodes found")
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be between zero and one")
    shuffled = ordered.copy()
    random.Random(seed).shuffle(shuffled)
    heldout_count = max(1, round(len(shuffled) * heldout_fraction))
    heldout = sorted(shuffled[:heldout_count])
    train = sorted(shuffled[heldout_count:])
    return train, heldout


def phase_labels(
    actions: np.ndarray, horizon: int = 40
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Infer horizon-aligned manipulation phases from left-hand closure."""
    if actions.ndim != 2 or actions.shape[1] < 29:
        raise ValueError("expected action array with at least 29 Unitree G1 joints")
    if len(actions) < horizon:
        raise ValueError("episode is shorter than the action horizon")

    left_hand = actions[:, 22:29]
    closure = np.linalg.norm(left_hand - left_hand[0], axis=1)
    peak = float(closure.max())
    threshold = max(0.4, 0.45 * peak)
    active = np.flatnonzero(closure > threshold)
    if not len(active):
        raise ValueError("could not detect a left-hand grasp event")
    close_at = int(active[0])
    release_at = int(active[-1]) + 1

    labels = np.empty(len(actions), dtype=np.int64)
    for step in range(len(actions)):
        end = step + horizon - 1
        if end < close_at:
            phase = 0
        elif step < close_at:
            phase = 1
        elif end < release_at:
            phase = 2
        else:
            phase = 3
        labels[step] = phase
    return labels, {
        "close_at": close_at,
        "release_at": release_at,
        "closure_peak": peak,
        "closure_threshold": threshold,
    }


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def materialize_split(
    source: Path,
    destination: Path,
    episode_rows: list[dict],
    original_ids: list[int],
    *,
    recovery_labels: bool,
    horizon: int,
) -> list[dict]:
    """Write one dense-ID LeRobot split and return its audit records."""
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing split: {destination}")
    (destination / "meta").mkdir(parents=True)
    metadata_by_id = {int(row["episode_index"]): row for row in episode_rows}
    labels = RECOVERY_SUBTASKS if recovery_labels else SUBTASKS
    records: list[dict] = []
    dense_episode_rows: list[dict] = []
    global_index = 0

    for dense_id, original_id in enumerate(original_ids):
        source_parquet = source / f"data/chunk-000/episode_{original_id:06d}.parquet"
        source_video = (
            source
            / "videos/chunk-000/observation.images.ego_view"
            / f"episode_{original_id:06d}.mp4"
        )
        if not source_parquet.is_file() or not source_video.is_file():
            raise FileNotFoundError(f"episode {original_id} is incomplete")

        frame = pd.read_parquet(source_parquet)
        actions = np.stack(frame["action"].to_numpy())
        task_indices, boundary = phase_labels(actions, horizon=horizon)
        frame["episode_index"] = dense_id
        frame["task_index"] = task_indices
        if "annotation.human.task_description" in frame:
            frame["annotation.human.task_description"] = task_indices
        frame["index"] = np.arange(global_index, global_index + len(frame), dtype=np.int64)
        global_index += len(frame)

        target_parquet = destination / f"data/chunk-000/episode_{dense_id:06d}.parquet"
        target_parquet.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target_parquet, index=False)
        target_video = (
            destination
            / "videos/chunk-000/observation.images.ego_view"
            / f"episode_{dense_id:06d}.mp4"
        )
        _link_or_copy(source_video, target_video)

        dense_episode_rows.append(
            {
                **metadata_by_id[original_id],
                "episode_index": dense_id,
                "tasks": list(labels),
            }
        )
        counts = np.bincount(task_indices, minlength=len(labels))
        records.append(
            {
                "episode_index": dense_id,
                "original_episode_index": original_id,
                "length": len(frame),
                "phase_frame_counts": counts.tolist(),
                **boundary,
            }
        )

    source_meta = source / "meta"
    info = json.loads((source_meta / "info.json").read_text(encoding="utf-8"))
    info.update(
        {
            "total_episodes": len(original_ids),
            "total_frames": sum(row["length"] for row in dense_episode_rows),
            "total_tasks": len(labels),
            "total_videos": len(original_ids),
            "total_chunks": 1,
            "splits": {"train": "0:100"},
        }
    )
    (destination / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    (destination / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in dense_episode_rows)
    )
    (destination / "meta/tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": i, "task": task}) + "\n" for i, task in enumerate(labels))
    )
    shutil.copy2(source_meta / "modality.json", destination / "meta/modality.json")
    # The source statistics let tools inspect both splits immediately.  The
    # training launcher recomputes and records train-only statistics.
    shutil.copy2(source_meta / "stats.json", destination / "meta/stats.json")
    relative = source_meta / "relative_stats.json"
    if relative.exists():
        shutil.copy2(relative, destination / "meta/relative_stats.json")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--recovery-labels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_rows = [
        json.loads(line)
        for line in (args.source / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_ids, heldout_ids = split_episode_ids(
        (int(row["episode_index"]) for row in episode_rows), args.heldout_fraction, args.seed
    )
    all_records: dict[str, list[dict]] = {}
    for split, ids in (("train", train_ids), ("heldout", heldout_ids)):
        all_records[split] = materialize_split(
            args.source,
            args.output_root / split,
            episode_rows,
            ids,
            recovery_labels=args.recovery_labels,
            horizon=args.horizon,
        )

    manifest = {
        "schema_version": 1,
        "source": str(args.source),
        "seed": args.seed,
        "heldout_fraction": args.heldout_fraction,
        "horizon": args.horizon,
        "recovery_labels": args.recovery_labels,
        "subtasks": list(RECOVERY_SUBTASKS if args.recovery_labels else SUBTASKS),
        "splits": all_records,
    }
    (args.output_root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "train_episodes": len(train_ids),
                "heldout_episodes": len(heldout_ids),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
