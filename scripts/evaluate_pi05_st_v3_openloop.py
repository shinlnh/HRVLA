#!/usr/bin/env python3
"""Paired, task-matched PI0.5 validation probe on the audited HA ST v3 view.

This is an open-loop first-action MSE diagnostic, not a closed-loop simulator
success metric. It only reads the validation split and never uses heldout data.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data._utils.collate import default_collate

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies import factory as policy_factory

from lerobot_pi05_low_mem_train import install_streaming_loader


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_episodes(root: Path, source_dataset: str) -> list[dict]:
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "meta/hrvla_export_audit.json").read_text(encoding="utf-8"))
    provenance = root / "meta/hrvla_source_episodes.jsonl"
    if (info.get("codebase_version") != "v3.0"
            or info.get("hrvla_subtask_relabel", {}).get("split") != "validation"
            or audit.get("status") != "loader_pass"
            or audit.get("provenance_sha256") != sha256(provenance)):
        raise ValueError("validation v3 view lacks a passing loader/provenance audit")
    rows = [json.loads(line) for line in provenance.read_text(encoding="utf-8").splitlines() if line]
    chosen = [row for row in rows if row["source_dataset"].split("/", 1)[0] == source_dataset]
    if len(chosen) != 10:
        raise ValueError(f"expected exactly 10 {source_dataset} validation episodes, got {len(chosen)}")
    return chosen


def sample_indices(rows: list[dict], frames_per_episode: int) -> tuple[list[int], list[int]]:
    indices: list[int] = []
    episode_ids: list[int] = []
    offset = 0
    for row in rows:
        length = int(row["length"])
        # Avoid boundary padding in the 20-action training target window.
        available = max(1, length - 20)
        for ordinal in range(frames_per_episode):
            frame = min(available - 1, (ordinal * available + available // 2) // frames_per_episode)
            indices.append(offset + frame)
            episode_ids.append(int(row["episode_index"]))
        offset += length
    return indices, episode_ids


def evaluate_one(
    checkpoint: Path,
    dataset: LeRobotDataset,
    train_stats: dict,
    tokenizer_dir: Path,
    indices: list[int],
    episode_ids: list[int],
    batch_size: int,
) -> dict:
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path = str(checkpoint)
    cfg.device = "cuda"
    # PI0.5's saved config requests max-autotune compilation. Eager inference
    # evaluates the same weights without a many-minute compilation and avoids
    # a CUDA device-side assert observed in the compiled validation path.
    cfg.compile_model = False
    model = policy_factory.make_policy(cfg=cfg, ds_meta=dataset.meta)
    model.eval()
    features = {**model.config.input_features, **model.config.output_features}
    preprocessor, postprocessor = policy_factory.make_pre_post_processors(
        policy_cfg=model.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "normalizer_processor": {
                "stats": train_stats,
                "features": features,
                "norm_map": model.config.normalization_mapping,
            },
            "tokenizer_processor": {"tokenizer_name": str(tokenizer_dir)},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": train_stats,
                "features": model.config.output_features,
                "norm_map": model.config.normalization_mapping,
            }
        },
    )
    per_episode: dict[int, list[float]] = {episode: [] for episode in sorted(set(episode_ids))}
    try:
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            raw = default_collate([dataset[idx] for idx in batch_indices])
            truth = raw["action"][:, 0, :].float().cpu()
            processed = preprocessor(raw)
            # Same noise and same frames for base and tuned checkpoint.
            torch.manual_seed(20260922 + start)
            torch.cuda.manual_seed_all(20260922 + start)
            with torch.inference_mode():
                action_chunk = model.predict_action_chunk(processed)
                predicted = postprocessor(action_chunk)[:, 0, :].float().cpu()
            if predicted.shape != truth.shape:
                raise ValueError(f"predicted {tuple(predicted.shape)} != truth {tuple(truth.shape)}")
            losses = (predicted - truth).square().mean(dim=1).tolist()
            for episode, loss in zip(episode_ids[start : start + len(losses)], losses, strict=True):
                per_episode[episode].append(float(loss))
            print(f"HRVLA: {checkpoint.name} evaluated {min(start + batch_size, len(indices))}/{len(indices)} frames", flush=True)
        means = {str(episode): sum(values) / len(values) for episode, values in per_episode.items()}
        return {
            "checkpoint": str(checkpoint),
            "checkpoint_config_sha256": sha256(checkpoint / "config.json"),
            "checkpoint_weights_sha256": sha256(checkpoint / "model.safetensors"),
            "macro_episode_first_action_mse": sum(means.values()) / len(means),
            "episode_first_action_mse": means,
            "frames_per_episode": len(indices) // len(means),
            "frames_total": len(indices),
        }
    finally:
        del preprocessor, postprocessor, model
        gc.collect()
        # Do not mask the real evaluation error if the CUDA context is poisoned.
        if torch.cuda.is_initialized():
            try:
                torch.cuda.empty_cache()
            except torch.AcceleratorError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument("--tuned-policy", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-episode", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.frames_per_episode < 1 or args.batch_size < 1 or args.output.exists():
        raise ValueError("positive sampling dimensions and a fresh output path are required")
    if args.source_dataset not in args.base_policy.parts:
        raise ValueError("base policy is not task matched")
    if os.environ.get("HRVLA_PI05_TOKENIZER_DIR") != str(args.tokenizer_dir):
        raise ValueError("set HRVLA_PI05_TOKENIZER_DIR to the pinned tokenizer")
    rows = selected_episodes(args.validation, args.source_dataset)
    indices, episode_ids = sample_indices(rows, args.frames_per_episode)
    install_streaming_loader()
    cfg = PreTrainedConfig.from_pretrained(args.base_policy)
    meta = LeRobotDatasetMetadata("local/validation", root=args.validation)
    dataset = LeRobotDataset(
        "local/validation", root=args.validation,
        episodes=[int(row["episode_index"]) for row in rows],
        delta_timestamps=resolve_delta_timestamps(cfg, meta),
    )
    if len(dataset) != sum(int(row["length"]) for row in rows):
        raise ValueError("filtered LeRobot validation lengths differ from provenance")
    train_meta = LeRobotDatasetMetadata("local/train", root=args.train)
    result = {
        "schema_version": 1,
        "metric": "open_loop_first_action_mse_macro_episode",
        "claim_boundary": "fixed-seed validation diagnostic only; not HA simulator SR, not heldout or paper-complete",
        "source_dataset": args.source_dataset,
        "validation_provenance_sha256": sha256(args.validation / "meta/hrvla_source_episodes.jsonl"),
        "train_stats_sha256": sha256(args.train / "meta/stats.json"),
        "episode_indices": [int(row["episode_index"]) for row in rows],
        "sample_local_indices": indices,
        "seed_rule": "20260922 + sample batch start index; paired base/tuned",
    }
    result["base"] = evaluate_one(args.base_policy, dataset, train_meta.stats, args.tokenizer_dir,
                                  indices, episode_ids, args.batch_size)
    print(f"HRVLA: base validation MSE={result['base']['macro_episode_first_action_mse']:.6f}", flush=True)
    result["tuned"] = evaluate_one(args.tuned_policy, dataset, train_meta.stats, args.tokenizer_dir,
                                   indices, episode_ids, args.batch_size)
    result["tuned_minus_base_mse"] = (
        result["tuned"]["macro_episode_first_action_mse"]
        - result["base"]["macro_episode_first_action_mse"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("metric", "source_dataset", "tuned_minus_base_mse")}, indent=2))
    print(f"base={result['base']['macro_episode_first_action_mse']:.6f} "
          f"tuned={result['tuned']['macro_episode_first_action_mse']:.6f}")


if __name__ == "__main__":
    main()
