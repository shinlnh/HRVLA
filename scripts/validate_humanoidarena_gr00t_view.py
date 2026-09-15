#!/usr/bin/env python3
"""Audit a materialized HumanoidArena GR00T view and sample exact video pixels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
    ACTION_FIELDS,
    STATE_FIELDS,
    canonical_json_sha256,
    modality_metadata,
)
from humanoidarena_gr00t_video import install_packed_video_offset_patch  # noqa: E402


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _load_gr00t_config(config_path: Path):
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag

    spec = importlib.util.spec_from_file_location("hrvla_ha_refpose_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import modality config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]


def _actual_storage_bytes(root: Path) -> int:
    return sum(
        path.lstat().st_size
        for path in root.rglob("*")
        if not path.is_dir() and not path.is_symlink()
    )


def validate_view(view_root: Path, config_path: Path) -> dict[str, Any]:
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.utils.video_utils import get_frames_by_indices

    manifest_path = view_root / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_modality_hash = canonical_json_sha256(modality_metadata())
    if manifest.get("modality_sha256") != expected_modality_hash:
        raise ValueError("view modality hash does not match the bridge implementation")
    config = _load_gr00t_config(config_path)
    if config["state"].modality_keys != [field.name for field in STATE_FIELDS]:
        raise ValueError("GR00T state groups differ from the bridge layout")
    if config["action"].modality_keys != [field.name for field in ACTION_FIELDS]:
        raise ValueError("GR00T action groups differ from the bridge layout")
    if config["action"].delta_indices != list(range(40)):
        raise ValueError("GR00T action horizon differs from the frozen 40-frame contract")

    expected: dict[tuple[str, str], set[int]] = {}
    for source in manifest["sources"]:
        dataset = str(source["dataset"])
        expected[("train", dataset)] = set(map(int, source["train_episode_ids"]))
        expected[("heldout", dataset)] = set(map(int, source["heldout_episode_ids"]))

    install_packed_video_offset_patch()
    observed: dict[tuple[str, str], set[int]] = {key: set() for key in expected}
    task_rows: dict[str, dict[str, Any]] = {}
    pixel_samples: list[dict[str, Any]] = []
    checked_lowdim = 0
    symlink_count = 0

    for split in ("train", "heldout"):
        split_root = view_root / split
        loader = LeRobotEpisodeLoader(split_root, config)
        if [int(row["episode_index"]) for row in loader.episodes_metadata] != list(
            range(len(loader.episodes_metadata))
        ):
            raise ValueError(f"{split}: episode IDs are not dense and ordered")

        samples: dict[str, int] = {}
        for dense_id, row in enumerate(loader.episodes_metadata):
            dataset = str(row["source_dataset"])
            source_episode = int(row["source_episode_index"])
            key = (split, dataset)
            if key not in observed:
                raise ValueError(f"unplanned source dataset in {split}: {dataset}")
            if source_episode in observed[key]:
                raise ValueError(f"duplicate source episode in {split}/{dataset}: {source_episode}")
            observed[key].add(source_episode)
            video_path = (
                split_root
                / "videos/chunk-000/observation.images.front"
                / f"episode_{dense_id:06d}.mp4"
            )
            if not video_path.is_symlink() or video_path.resolve() != Path(row["source_video"]):
                raise ValueError(f"invalid packed-video symlink: {video_path}")
            symlink_count += 1
            task = task_rows.setdefault(
                dataset,
                {
                    "task": dataset.split("/", 1)[0],
                    "train_episodes": 0,
                    "heldout_episodes": 0,
                    "train_frames": 0,
                    "heldout_frames": 0,
                },
            )
            task[f"{split}_episodes"] += 1
            task[f"{split}_frames"] += int(row["length"])
            if int(row["video_frame_offset"]) > 0 and dataset not in samples:
                samples[dataset] = dense_id

        for dataset, dense_id in sorted(samples.items()):
            row = loader.episodes_metadata[dense_id]
            length = int(row["length"])
            local_index = length // 2
            packed_index = int(row["video_frame_offset"]) + local_index
            local = np.asarray([local_index], dtype=np.int64)
            through_view = loader._load_video_data(dense_id, local)["front"]
            direct = get_frames_by_indices(
                str(row["source_video"]),
                np.asarray([packed_index], dtype=np.int64),
                decoder_kwargs={},
            )
            if not np.array_equal(through_view, direct):
                raise ValueError(f"pixel mismatch for {split}/{dataset}/{dense_id}")

            raw_path = split_root / f"data/chunk-000/episode_{dense_id:06d}.parquet"
            raw = pd.read_parquet(raw_path)
            grouped = loader._load_parquet_data(dense_id)
            for index in (0, length - 1):
                reconstructed_state = np.concatenate(
                    [
                        np.asarray(grouped[f"state.{field.name}"].iloc[index])
                        for field in STATE_FIELDS
                    ]
                )
                reconstructed_action = np.concatenate(
                    [
                        np.asarray(grouped[f"action.{field.name}"].iloc[index])
                        for field in ACTION_FIELDS
                    ]
                )
                if not np.array_equal(
                    reconstructed_state, np.asarray(raw["observation.state"].iloc[index])
                ):
                    raise ValueError(f"state round-trip mismatch for {split}/{dataset}/{dense_id}")
                if not np.array_equal(
                    reconstructed_action, np.asarray(raw["action"].iloc[index])
                ):
                    raise ValueError(f"action round-trip mismatch for {split}/{dataset}/{dense_id}")
                checked_lowdim += 1
            pixel_samples.append(
                {
                    "split": split,
                    "source_dataset": dataset,
                    "source_episode_index": int(row["source_episode_index"]),
                    "local_frame_index": local_index,
                    "packed_frame_index": packed_index,
                    "pixel_sha256": _sha256_bytes(through_view.tobytes()),
                }
            )

    if observed != expected:
        differences = {
            f"{split}/{dataset}": {
                "missing": sorted(expected[(split, dataset)] - observed[(split, dataset)]),
                "extra": sorted(observed[(split, dataset)] - expected[(split, dataset)]),
            }
            for split, dataset in expected
            if observed[(split, dataset)] != expected[(split, dataset)]
        }
        raise ValueError(f"view membership differs from manifest: {differences}")
    for dataset in {dataset for _, dataset in expected}:
        if expected[("train", dataset)] & expected[("heldout", dataset)]:
            raise ValueError(f"train/held-out leakage in manifest: {dataset}")

    totals = {
        split: {
            "episodes": sum(row[f"{split}_episodes"] for row in task_rows.values()),
            "frames": sum(row[f"{split}_frames"] for row in task_rows.values()),
        }
        for split in ("train", "heldout")
    }
    if totals != manifest["outputs"]:
        raise ValueError(f"view totals differ from manifest: {totals} != {manifest['outputs']}")
    return {
        "schema_version": 1,
        "status": "pass",
        "claim_boundary": (
            "dataset/action-interface validation only; not policy quality, task success, "
            "or recovery"
        ),
        "source_revision": manifest["source_revision"],
        "view_manifest_sha256": _sha256_file(manifest_path),
        "modality_sha256": expected_modality_hash,
        "state_widths": {field.name: field.width for field in STATE_FIELDS},
        "action_widths": {field.name: field.width for field in ACTION_FIELDS},
        "action_horizon": 40,
        "totals": totals,
        "tasks": [task_rows[key] for key in sorted(task_rows)],
        "checks": {
            "all_episode_memberships_match_manifest": True,
            "train_heldout_disjoint_per_task": True,
            "dense_episode_ids": True,
            "packed_video_symlinks": symlink_count,
            "pixel_exact_samples": len(pixel_samples),
            "lowdim_roundtrip_rows": checked_lowdim,
            "actual_storage_bytes_excluding_symlink_targets": _actual_storage_bytes(view_root),
        },
        "pixel_samples": pixel_samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument(
        "--modality-config",
        type=Path,
        default=ROOT / "config/g1_humanoidarena_refpose_config.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_view(args.view_root.resolve(), args.modality_config.resolve())
    _write_json_atomic(args.output.resolve(), report)
    print(json.dumps({"status": report["status"], **report["checks"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
