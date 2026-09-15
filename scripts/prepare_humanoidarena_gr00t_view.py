#!/usr/bin/env python3
"""Create leakage-resistant Isaac-GR00T views of HumanoidArena LeRobot-v3 data.

Low-dimensional rows are materialized per episode in the v2 layout expected by
Isaac-GR00T.  Videos are never copied or re-encoded: each episode path is a
symlink to its immutable packed v3 MP4 and ``video_frame_offset`` tells the
runtime shim where that episode begins.  This preserves pixels while avoiding
another roughly 10 GB local copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.humanoidarena_bridge import (  # noqa: E402
    canonical_json_sha256,
    deterministic_split,
    deterministic_train_validation_hidden_split,
    modality_metadata,
    validate_release_info,
)


DATASET_REVISION = "a079beddd6b1521f762c991be8f36993f17ebeca"
VIEW_NAME = "sonic_refpose_v3_1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tasks(path: Path) -> list[str]:
    frame = pd.read_parquet(path)
    if frame.index.name == "task":
        return [str(value) for value in frame.index.tolist()]
    if "task" in frame.columns:
        return [str(value) for value in frame["task"].tolist()]
    raise ValueError(f"tasks parquet has no task text: {path}")


def _episode_table(dataset: Path) -> pd.DataFrame:
    paths = sorted((dataset / "meta/episodes").glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no episode metadata parquet under {dataset}")
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def _source_file(dataset: Path, pattern: str, chunk: int, file_index: int, **extra: Any) -> Path:
    values = {
        "chunk_index": chunk,
        "file_index": file_index,
        "episode_chunk": chunk,
        "episode_index": file_index,
        **extra,
    }
    return dataset / pattern.format(**values)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _stats(frames: list[pd.DataFrame]) -> dict[str, Any]:
    if not frames:
        raise ValueError("cannot compute statistics for an empty split")
    output: dict[str, Any] = {}
    for key in ("observation.state", "action"):
        values = np.vstack(
            [
                np.vstack(frame[key].map(lambda value: np.asarray(value, dtype=np.float32)))
                for frame in frames
            ]
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


def _prepare_split_dirs(output_root: Path, split_names: tuple[str, ...]) -> dict[str, Path]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing view: {output_root}")
    roots = {name: output_root / name for name in split_names}
    for root in roots.values():
        (root / "meta").mkdir(parents=True)
    return roots


def materialize_view(
    source_root: Path,
    output_root: Path,
    *,
    heldout_fraction: float,
    seed: int,
    validation_fraction: float = 0.0,
) -> dict[str, Any]:
    split_names = (
        ("train", "validation", "heldout")
        if validation_fraction > 0.0
        else ("train", "heldout")
    )
    split_roots = _prepare_split_dirs(output_root, split_names)
    episode_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in split_roots}
    split_frames: dict[str, list[pd.DataFrame]] = {name: [] for name in split_roots}
    task_rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    dense_ids = {name: 0 for name in split_roots}
    global_indices = {name: 0 for name in split_roots}
    template_info: dict[str, Any] | None = None

    datasets = sorted(path for path in source_root.glob(f"*/{VIEW_NAME}") if path.is_dir())
    if not datasets:
        raise FileNotFoundError(f"no */{VIEW_NAME} datasets under {source_root}")

    for task_index, dataset in enumerate(datasets):
        info_path = dataset / "meta/info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        validate_release_info(info)
        template_info = template_info or info
        tasks = _load_tasks(dataset / "meta/tasks.parquet")
        if len(tasks) != 1:
            raise ValueError(f"expected one task per HumanoidArena directory: {dataset}")
        task_text = tasks[0]
        task_rows.append({"task_index": task_index, "task": task_text})
        episodes = _episode_table(dataset)
        source_ids = [int(value) for value in episodes["episode_index"].tolist()]
        if validation_fraction > 0.0:
            train_ids, validation_ids, heldout_ids = (
                deterministic_train_validation_hidden_split(
                    source_ids,
                    validation_fraction,
                    heldout_fraction,
                    seed + task_index,
                )
            )
        else:
            train_ids, heldout_ids = deterministic_split(
                source_ids, heldout_fraction, seed + task_index
            )
            validation_ids = []
        membership = {value: "train" for value in train_ids}
        membership.update({value: "validation" for value in validation_ids})
        membership.update({value: "heldout" for value in heldout_ids})

        data_cache: dict[tuple[int, int], pd.DataFrame] = {}
        for _, metadata in episodes.sort_values("episode_index").iterrows():
            source_episode = int(metadata["episode_index"])
            split = membership[source_episode]
            destination = split_roots[split]
            dense_id = dense_ids[split]
            dense_ids[split] += 1
            data_chunk = int(metadata["data/chunk_index"])
            data_file = int(metadata["data/file_index"])
            cache_key = (data_chunk, data_file)
            if cache_key not in data_cache:
                path = _source_file(dataset, info["data_path"], data_chunk, data_file)
                data_cache[cache_key] = pd.read_parquet(path)
            frame = data_cache[cache_key]
            frame = frame[frame["episode_index"] == source_episode].copy().reset_index(drop=True)
            declared_length = int(metadata["length"])
            if len(frame) != declared_length:
                raise ValueError(
                    f"{dataset}: episode {source_episode} has {len(frame)} rows, "
                    f"metadata declares {declared_length}"
                )
            frame["episode_index"] = dense_id
            frame["task_index"] = task_index
            frame["frame_index"] = np.arange(len(frame), dtype=np.int64)
            frame["index"] = np.arange(
                global_indices[split], global_indices[split] + len(frame), dtype=np.int64
            )
            global_indices[split] += len(frame)
            parquet = destination / f"data/chunk-000/episode_{dense_id:06d}.parquet"
            parquet.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(parquet, index=False)
            split_frames[split].append(frame)

            video_key = "observation.images.front"
            video_chunk = int(metadata[f"videos/{video_key}/chunk_index"])
            video_file = int(metadata[f"videos/{video_key}/file_index"])
            video_source = _source_file(
                dataset,
                info["video_path"],
                video_chunk,
                video_file,
                video_key=video_key,
            ).resolve()
            if not video_source.is_file():
                raise FileNotFoundError(video_source)
            video_destination = (
                destination
                / "videos/chunk-000/observation.images.front"
                / f"episode_{dense_id:06d}.mp4"
            )
            video_destination.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(video_source, video_destination)
            fps = float(info["fps"])
            from_timestamp = float(metadata[f"videos/{video_key}/from_timestamp"])
            to_timestamp = float(metadata[f"videos/{video_key}/to_timestamp"])
            video_offset = int(round(from_timestamp * fps))
            available_frames = int(round((to_timestamp - from_timestamp) * fps))
            if available_frames < len(frame):
                raise ValueError(
                    f"{dataset}: episode {source_episode} video exposes "
                    f"{available_frames} frames for {len(frame)} rows"
                )
            episode_rows[split].append(
                {
                    "episode_index": dense_id,
                    "tasks": [task_text],
                    "length": len(frame),
                    "video_frame_offset": video_offset,
                    "source_dataset": str(dataset.relative_to(source_root)),
                    "source_episode_index": source_episode,
                    "source_video": str(video_source),
                }
            )

        source_records.append(
            {
                "dataset": str(dataset.relative_to(source_root)),
                "info_sha256": _sha256(info_path),
                "episodes": len(source_ids),
                "train_episode_ids": train_ids,
                **(
                    {"validation_episode_ids": validation_ids}
                    if validation_fraction > 0.0
                    else {}
                ),
                "heldout_episode_ids": heldout_ids,
            }
        )

    assert template_info is not None
    for split, destination in split_roots.items():
        features = dict(template_info["features"])
        info = {
            "codebase_version": "v2.1",
            "robot_type": "unitree_g1_refpose_v3_1",
            "total_episodes": len(episode_rows[split]),
            "total_frames": sum(row["length"] for row in episode_rows[split]),
            "total_tasks": len(task_rows),
            "total_videos": len(episode_rows[split]),
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": int(template_info["fps"]),
            "splits": {"train": "0:100"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/"
                "episode_{episode_index:06d}.mp4"
            ),
            "features": features,
            "vla_protocol": template_info["vla_protocol"],
            "hrvla_view": {
                "schema_version": 1,
                "source_revision": DATASET_REVISION,
                "packed_video_offsets": True,
                "split": split,
            },
        }
        _write_json(destination / "meta/info.json", info)
        _write_json(destination / "meta/modality.json", modality_metadata())
        _write_json(destination / "meta/stats.json", _stats(split_frames[split]))
        _write_jsonl(destination / "meta/tasks.jsonl", task_rows)
        _write_jsonl(destination / "meta/episodes.jsonl", episode_rows[split])

    manifest = {
        "schema_version": 2 if validation_fraction > 0.0 else 1,
        "source_revision": DATASET_REVISION,
        "source_root": str(source_root.resolve()),
        "heldout_fraction": heldout_fraction,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "split_policy": (
            "deterministic per task before any training or evaluation; hidden selected "
            "first, validation second, remaining episodes used for training"
        ),
        "split_roles": {
            "train": "parameter optimization",
            **(
                {"validation": "checkpoint and hyperparameter selection"}
                if validation_fraction > 0.0
                else {}
            ),
            "heldout": "hidden final test; forbidden for checkpoint or hyperparameter selection",
        },
        "video_policy": "immutable packed MP4 symlinks plus exact frame offsets; no re-encode",
        "modality_sha256": canonical_json_sha256(modality_metadata()),
        "sources": source_records,
        "outputs": {
            split: {
                "episodes": len(episode_rows[split]),
                "frames": sum(row["length"] for row in episode_rows[split]),
            }
            for split in split_roots
        },
    }
    _write_json(output_root / "split_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    manifest = materialize_view(
        args.source_root.resolve(),
        args.output_root.resolve(),
        heldout_fraction=args.heldout_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(json.dumps(manifest["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
