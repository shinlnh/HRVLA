#!/usr/bin/env python3
"""Export audited ST labels to LeRobot v3 without re-encoding packed videos.

The v2.1 ST view contains per-episode Parquet files but symlinks each episode
to an entire v3 source video shard. The stock converter treats each symlink as
one complete episode and would duplicate videos and lose frame alignment.
This exporter hard-links the immutable data/video bytes into a new v3 view and
reconstructs per-episode offsets from the original v3 metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import pandas as pd
import pyarrow.parquet as pq


VIDEO_KEY = "observation.images.front"
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _source_metadata(reference_root: Path, relative: str, cache: dict) -> dict:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"invalid source dataset path: {relative}")
    if relative not in cache:
        root = (reference_root / relative_path).resolve()
        if not _under(root, reference_root.resolve()):
            raise ValueError(f"source dataset escapes reference root: {relative}")
        info_path = root / "meta/info.json"
        info = _json(info_path)
        if info.get("codebase_version") != "v3.0":
            raise ValueError(f"reference dataset is not LeRobot v3: {root}")
        episode_files = sorted((root / "meta/episodes").glob("*/*.parquet"))
        if not episode_files:
            raise FileNotFoundError(f"reference episode metadata missing: {root}")
        episodes = {}
        for path in episode_files:
            for row in pq.read_table(path).to_pylist():
                index = int(row["episode_index"])
                if index in episodes:
                    raise ValueError(f"duplicate reference episode {relative}:{index}")
                episodes[index] = row
        cache[relative] = {"root": root, "info": info, "episodes": episodes,
                           "info_sha256": _sha256(info_path)}
    return cache[relative]


def plan_split(source_root: Path, reference_root: Path, split: str) -> dict:
    """Audit every mapping before writing any output files."""
    if split not in {"train", "validation"}:
        raise ValueError("only train/validation can be exported before model selection")
    source = source_root / split
    info_path = source / "meta/info.json"
    info = _json(info_path)
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"expected audited ST v2.1 view at {source}")
    features = info.get("features", {})
    if features.get("observation.state", {}).get("shape") != [64]:
        raise ValueError("state must be 64-D")
    if features.get("action", {}).get("shape") != [40]:
        raise ValueError("action must be 40-D")
    fps = float(info["fps"])
    if fps <= 0:
        raise ValueError("fps must be positive")
    tasks = _jsonl(source / "meta/tasks.jsonl")
    task_map = {int(row["task_index"]): str(row["task"]) for row in tasks}
    if len(task_map) != len(tasks) or set(task_map) != set(range(len(tasks))):
        raise ValueError("task indices must be unique and contiguous")
    rows = _jsonl(source / "meta/episodes.jsonl")
    if len(rows) != int(info["total_episodes"]):
        raise ValueError("episode count differs from info.json")

    cache: dict = {}
    video_ids: dict[Path, int] = {}
    episodes = []
    total_frames = 0
    for expected_index, row in enumerate(rows):
        index = int(row["episode_index"])
        length = int(row["length"])
        if index != expected_index or length <= 0:
            raise ValueError(f"invalid episode order/length at {expected_index}")
        parquet = source / info["data_path"].format(
            episode_chunk=index // int(info["chunks_size"]), episode_index=index
        )
        if not parquet.is_file() or pq.read_metadata(parquet).num_rows != length:
            raise ValueError(f"Parquet length mismatch: {parquet}")
        small = pq.read_table(
            parquet, columns=["episode_index", "frame_index", "index", "task_index"]
        ).to_pydict()
        if (set(small["episode_index"]) != {index}
                or small["frame_index"] != list(range(length))
                or small["index"] != list(range(total_frames, total_frames + length))):
            raise ValueError(f"frame/episode/global index mismatch: {parquet}")
        permitted_tasks = set(row["tasks"])
        if not permitted_tasks or any(task_map.get(int(key)) not in permitted_tasks
                                      for key in small["task_index"]):
            raise ValueError(f"sub-task labels are not in episode program: {index}")

        reference = _source_metadata(reference_root, str(row["source_dataset"]), cache)
        original = reference["episodes"].get(int(row["source_episode_index"]))
        if original is None or int(original["length"]) != length:
            raise ValueError(f"source v3 episode/length mismatch: {index}")
        source_video = Path(row["source_video"]).resolve(strict=True)
        if not _under(source_video, reference_root.resolve()):
            raise ValueError(f"video escapes reference root: {source_video}")
        expected_video = reference["root"] / reference["info"]["video_path"].format(
            video_key=VIDEO_KEY,
            chunk_index=int(original[f"videos/{VIDEO_KEY}/chunk_index"]),
            file_index=int(original[f"videos/{VIDEO_KEY}/file_index"]),
        )
        if source_video != expected_video.resolve(strict=True):
            raise ValueError(f"video shard differs from reference metadata: {index}")
        video_offset = int(row["video_frame_offset"])
        start = float(original[f"videos/{VIDEO_KEY}/from_timestamp"])
        end = float(original[f"videos/{VIDEO_KEY}/to_timestamp"])
        if abs(start * fps - video_offset) > 0.5 or abs(end * fps - video_offset - length) > 0.5:
            raise ValueError(f"packed-video frame offset mismatch: {index}")
        if source_video not in video_ids:
            video_ids[source_video] = len(video_ids)
        episodes.append({
            "episode_index": index, "tasks": list(row["tasks"]), "length": length,
            "data/chunk_index": 0, "data/file_index": index,
            "dataset_from_index": total_frames,
            "dataset_to_index": total_frames + length,
            f"videos/{VIDEO_KEY}/chunk_index": 0,
            f"videos/{VIDEO_KEY}/file_index": video_ids[source_video],
            f"videos/{VIDEO_KEY}/from_timestamp": start,
            f"videos/{VIDEO_KEY}/to_timestamp": end,
            "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
            "_source_parquet": parquet,
        })
        total_frames += length
    if total_frames != int(info["total_frames"]):
        raise ValueError("total frame count differs from info.json")
    return {"source": source, "info": info, "tasks": tasks, "source_rows": rows,
            "episodes": episodes,
            "videos": video_ids, "references": cache, "total_frames": total_frames,
            "source_info_sha256": _sha256(info_path),
            "source_episodes_sha256": _sha256(source / "meta/episodes.jsonl")}


def export_split(plan: dict, output_root: Path, split: str) -> Path:
    """Create an independent v3 view with hard links; never mutate the source."""
    target = output_root / split
    building = output_root / f".{split}.building"
    if target.exists() or building.exists():
        raise FileExistsError(f"refusing to overwrite {target} or {building}")
    output_root.mkdir(parents=True, exist_ok=True)
    building.mkdir(parents=True)
    source = plan["source"]
    try:
        info = dict(plan["info"])
        info["codebase_version"] = "v3.0"
        info["data_path"] = DATA_PATH
        info["video_path"] = VIDEO_PATH
        info["data_files_size_in_mb"] = 100
        info["video_files_size_in_mb"] = 200
        info["splits"] = {"train": f"0:{len(plan['episodes'])}"}
        info["hrvla_packed_video_export"] = {
            "schema_version": 1, "source_info_sha256": plan["source_info_sha256"],
            "source_episodes_sha256": plan["source_episodes_sha256"],
            "source_v3_info_sha256": {
                name: item["info_sha256"] for name, item in plan["references"].items()
            },
        }
        info.pop("total_chunks", None)
        info.pop("total_videos", None)
        (building / "meta").mkdir()
        (building / "meta/info.json").write_text(
            json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.copy2(source / "meta/stats.json", building / "meta/stats.json")
        shutil.copy2(
            source / "meta/episodes.jsonl", building / "meta/hrvla_source_episodes.jsonl"
        )
        modality = source / "meta/modality.json"
        if modality.is_file():
            shutil.copy2(modality, building / "meta/modality.json")
        task_frame = pd.DataFrame(
            [{"task": row["task"], "task_index": int(row["task_index"])}
             for row in plan["tasks"]]
        ).set_index("task")
        task_frame.to_parquet(building / "meta/tasks.parquet")

        metadata = []
        for episode in plan["episodes"]:
            row = dict(episode)
            source_parquet = row.pop("_source_parquet")
            destination = building / DATA_PATH.format(
                chunk_index=0, file_index=row["episode_index"]
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source_parquet, destination)
            metadata.append(row)
        episode_path = building / "meta/episodes/chunk-000/file-000.parquet"
        episode_path.parent.mkdir(parents=True)
        pd.DataFrame(metadata).to_parquet(episode_path, index=False)
        for path, file_index in plan["videos"].items():
            destination = building / VIDEO_PATH.format(
                video_key=VIDEO_KEY, chunk_index=0, file_index=file_index
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, destination)
        audit = {
            "schema_version": 1, "status": "structural_pass_loader_pending",
            "split": split, "episodes": len(metadata), "frames": plan["total_frames"],
            "unique_video_shards": len(plan["videos"]),
            "source_info_sha256": plan["source_info_sha256"],
            "source_episodes_sha256": plan["source_episodes_sha256"],
        }
        (building / "meta/hrvla_export_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(building, target)
    except BaseException:
        # Keep a failed build for inspection; never remove or overwrite source data.
        raise
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "both"], default="both")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    splits = ["train", "validation"] if args.split == "both" else [args.split]
    plans = {split: plan_split(args.source_root, args.reference_root, split) for split in splits}
    for split, plan in plans.items():
        print(json.dumps({"split": split, "episodes": len(plan["episodes"]),
                          "frames": plan["total_frames"],
                          "video_shards": len(plan["videos"]), "audit_only": args.audit_only}))
    if not args.audit_only:
        for split, plan in plans.items():
            print(f"exported {export_split(plan, args.output_root, split)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
