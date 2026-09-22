#!/usr/bin/env python3
"""Audit a packed-video ST v3 export before any PI0.5 fine-tuning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from export_humanoidarena_st_v3 import DATA_PATH, VIDEO_KEY, VIDEO_PATH, _sha256, plan_split


def _same_inode(left: Path, right: Path) -> bool:
    a, b = left.stat(), right.stat()
    return (a.st_dev, a.st_ino, a.st_size) == (b.st_dev, b.st_ino, b.st_size)


def verify_split(plan: dict, output_root: Path, split: str, *, repair_stats: bool = False) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    target = output_root / split
    info = json.loads((target / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0" or (
        info.get("total_episodes"), info.get("total_frames")
    ) != (len(plan["episodes"]), plan["total_frames"]):
        raise ValueError(f"v3 info does not match audited source: {split}")
    lock = info.get("hrvla_packed_video_export", {})
    if lock.get("source_info_sha256") != plan["source_info_sha256"] or lock.get(
        "source_episodes_sha256"
    ) != plan["source_episodes_sha256"]:
        raise ValueError(f"v3 source lock mismatch: {split}")
    provenance = target / "meta/hrvla_source_episodes.jsonl"
    if not provenance.exists():
        # Supports v3 exports made before the provenance-file addition.
        shutil.copy2(plan["source"] / "meta/episodes.jsonl", provenance)
    if _sha256(provenance) != plan["source_episodes_sha256"]:
        raise ValueError(f"v3 source episode provenance mismatch: {split}")
    source_stats = json.loads((plan["source"] / "meta/stats.json").read_text(encoding="utf-8"))
    numeric_stats = {key: value for key, value in source_stats.items()
                     if key != "__fingerprints__"}
    stats_path = target / "meta/stats.json"
    exported_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if exported_stats != numeric_stats:
        if not repair_stats or exported_stats != source_stats:
            raise ValueError(f"v3 numeric stats differ from the audited source: {split}")
        temporary_stats = stats_path.with_suffix(".json.tmp")
        temporary_stats.write_text(
            json.dumps(numeric_stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_stats, stats_path)
    tasks = pd.read_parquet(target / "meta/tasks.parquet")
    if len(tasks) != len(plan["tasks"]):
        raise ValueError(f"v3 task count mismatch: {split}")
    actual = pd.read_parquet(target / "meta/episodes/chunk-000/file-000.parquet")
    if len(actual) != len(plan["episodes"]):
        raise ValueError(f"v3 episode metadata count mismatch: {split}")
    for expected, (_, row) in zip(plan["episodes"], actual.iterrows(), strict=True):
        for key in ("episode_index", "length", "data/chunk_index", "data/file_index",
                    "dataset_from_index", "dataset_to_index",
                    f"videos/{VIDEO_KEY}/chunk_index", f"videos/{VIDEO_KEY}/file_index",
                    "meta/episodes/chunk_index", "meta/episodes/file_index"):
            if int(row[key]) != int(expected[key]):
                raise ValueError(f"v3 episode metadata mismatch at {expected['episode_index']}:{key}")
        for key in (f"videos/{VIDEO_KEY}/from_timestamp",
                    f"videos/{VIDEO_KEY}/to_timestamp"):
            if abs(float(row[key]) - float(expected[key])) > 1e-6:
                raise ValueError(f"v3 video timestamp mismatch at {expected['episode_index']}:{key}")
        if list(row["tasks"]) != expected["tasks"]:
            raise ValueError(f"v3 episode task list mismatch at {expected['episode_index']}")
        parquet = target / DATA_PATH.format(chunk_index=0, file_index=expected["episode_index"])
        if not _same_inode(expected["_source_parquet"], parquet):
            raise ValueError(f"v3 data bytes differ from audited ST labels: {parquet}")
    for video, file_index in plan["videos"].items():
        destination = target / VIDEO_PATH.format(
            video_key=VIDEO_KEY, chunk_index=0, file_index=file_index
        )
        if not _same_inode(video, destination):
            raise ValueError(f"v3 video bytes differ from pinned reference: {destination}")

    dataset = LeRobotDataset(f"local/hrvla-st-v3-{split}", root=target)
    if len(dataset) != plan["total_frames"] or dataset.num_episodes != len(actual):
        raise ValueError(f"LeRobot loader frame/episode count mismatch: {split}")
    representative: dict[str, int] = {}
    for row in plan["source_rows"]:
        representative.setdefault(row["source_dataset"], int(row["episode_index"]))
    probes = []
    for source_name, episode_index in representative.items():
        episode = plan["episodes"][episode_index]
        old = pq.read_table(
            episode["_source_parquet"],
            columns=["observation.state", "action", "task_index"],
        ).to_pydict()
        offsets = (0, episode["length"] - 1) if split == "train" else (0,)
        for offset in offsets:
            sample = dataset[episode["dataset_from_index"] + offset]
            if not np.allclose(sample["observation.state"].numpy(),
                               old["observation.state"][offset]):
                raise ValueError(f"LeRobot state mismatch: {split}:{episode_index}:{offset}")
            if not np.allclose(sample["action"].numpy(), old["action"][offset]):
                raise ValueError(f"LeRobot action mismatch: {split}:{episode_index}:{offset}")
            if int(sample["task_index"]) != int(old["task_index"][offset]):
                raise ValueError(f"LeRobot task mismatch: {split}:{episode_index}:{offset}")
            if tuple(sample[f"observation.images.front"].shape) != (3, 480, 640):
                raise ValueError(f"LeRobot video decode mismatch: {split}:{episode_index}:{offset}")
            probes.append({"source_dataset": source_name, "episode_index": episode_index,
                           "frame_offset": offset})
    audit_path = target / "meta/hrvla_export_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.update({"status": "loader_pass", "loader_probes": probes,
                  "provenance_sha256": _sha256(provenance),
                  "numeric_stats_sha256": _sha256(stats_path),
                  "data_hardlinks_checked": len(plan["episodes"]),
                  "video_hardlinks_checked": len(plan["videos"])})
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, audit_path)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repair-stats", action="store_true",
                        help="remove only the nonnumeric v2 __fingerprints__ map")
    args = parser.parse_args()
    for split in ("train", "validation"):
        plan = plan_split(args.source_root, args.reference_root, split)
        audit = verify_split(plan, args.output_root, split, repair_stats=args.repair_stats)
        print(json.dumps({"split": split, "status": audit["status"],
                          "episodes": audit["episodes"], "frames": audit["frames"],
                          "loader_probes": len(audit["loader_probes"]) }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
