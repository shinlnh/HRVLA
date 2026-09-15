#!/usr/bin/env python3
"""Paired open-loop evaluation for in-domain GR00T VLA checkpoints."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import platform
import shutil
import statistics
import time

import numpy as np
import torch


CONDITIONS = ("clean", "vision_noise", "occlusion", "state_noise", "combined")


def _corrupt_observation(observation: dict, condition: str, rng: np.random.Generator) -> None:
    if condition in {"vision_noise", "combined"}:
        for key, value in observation["video"].items():
            noise = rng.normal(0.0, 20.0, value.shape)
            observation["video"][key] = np.clip(value.astype(np.float32) + noise, 0, 255).astype(
                np.uint8
            )
    if condition in {"occlusion", "combined"}:
        for value in observation["video"].values():
            height, width = value.shape[-3:-1]
            value[..., height // 3 : 2 * height // 3, width // 3 : 2 * width // 3, :] = 0
    if condition in {"state_noise", "combined"}:
        for key, value in observation["state"].items():
            observation["state"][key] = (
                value + rng.normal(0.0, 0.02, value.shape).astype(np.float32)
            ).astype(np.float32)


def _concatenate_columns(frame, columns: list[str]) -> np.ndarray:
    return np.concatenate([np.vstack(frame[column].to_numpy()) for column in columns], axis=-1)


def evaluate_trajectory(
    policy,
    loader,
    trajectory_id: int,
    condition: str,
    *,
    execution_horizon: int,
    steps: int,
    seed: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.utils import parse_observation_gr00t

    trajectory = loader[trajectory_id]
    actual_steps = min(steps, len(trajectory))
    state_keys = loader.modality_configs["state"].modality_keys
    action_keys = loader.modality_configs["action"].modality_keys
    input_config = deepcopy(loader.modality_configs)
    input_config.pop("action")
    predictions: list[np.ndarray] = []
    inference_latencies: list[float] = []

    for step in range(0, actual_steps, execution_horizon):
        point = extract_step_data(trajectory, step, input_config, policy.embodiment_tag)
        flat_observation = {
            **{f"video.{key}": np.asarray(value) for key, value in point.images.items()},
            **{
                f"state.{key}": np.asarray(value, dtype=np.float32)
                for key, value in point.states.items()
            },
            policy.language_key: point.text,
        }
        parsed = parse_observation_gr00t(flat_observation, loader.modality_configs)
        _corrupt_observation(parsed, condition, np.random.default_rng(seed + step))
        torch.manual_seed(seed + step)
        torch.cuda.manual_seed_all(seed + step)
        started = time.perf_counter()
        action, _ = policy.get_action(parsed)
        torch.cuda.synchronize()
        inference_latencies.append((time.perf_counter() - started) * 1_000)
        for offset in range(execution_horizon):
            predictions.append(
                np.concatenate(
                    [np.atleast_1d(np.asarray(action[key])[0][offset]) for key in action_keys]
                )
            )

    ground_truth = _concatenate_columns(
        trajectory, [f"action.{key}" for key in action_keys]
    )[:actual_steps]
    predicted = np.asarray(predictions)[:actual_steps]
    absolute_error = np.abs(ground_truth - predicted)
    squared_error = np.square(ground_truth - predicted)
    phase_text = np.asarray(trajectory[f"language.{policy.language_key}"].tolist())[:actual_steps]
    phase_metrics = {}
    for phase in sorted(set(phase_text.tolist())):
        mask = phase_text == phase
        phase_metrics[phase] = {
            "frames": int(mask.sum()),
            "mse": float(squared_error[mask].mean()),
            "mae": float(absolute_error[mask].mean()),
        }
    metrics = {
        "trajectory_id": trajectory_id,
        "condition": condition,
        "frames": actual_steps,
        "mse": float(squared_error.mean()),
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(squared_error.mean())),
        "p95_absolute_error": float(np.quantile(absolute_error, 0.95)),
        "within_0_1_rate": float((absolute_error < 0.1).mean()),
        "mean_inference_ms": statistics.fmean(inference_latencies),
        "p95_inference_ms": float(np.quantile(inference_latencies, 0.95)),
        "phase_metrics": phase_metrics,
    }
    arrays = {
        "ground_truth": ground_truth,
        "predicted": predicted,
        "phase_text": phase_text,
    }
    return metrics, arrays


def _write_charts(
    rows: list[dict],
    sample_arrays: dict[str, np.ndarray],
    sample_trajectory,
    output_dir: Path,
    video_key: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        if selected:
            grouped[condition] = {
                "mse": statistics.fmean(row["mse"] for row in selected),
                "mae": statistics.fmean(row["mae"] for row in selected),
                "within": statistics.fmean(row["within_0_1_rate"] for row in selected),
            }
    labels = list(grouped)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("mse", "mae", "within"),
        (
            "Action MSE (lower is better)",
            "Action MAE (lower is better)",
            "|error| < 0.1 (higher is better)",
        ),
    ):
        values = [grouped[label][metric] for label in labels]
        bars = axis.bar([label.replace("_", "\n") for label in labels], values, color="#2b6cb0")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.3f")
    figure.savefig(output_dir / "condition_metrics.png", dpi=180)
    plt.close(figure)

    ground_truth = sample_arrays["ground_truth"]
    predicted = sample_arrays["predicted"]
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    dimensions = (15, 18, 22)
    for axis, dimension in zip(axes, dimensions):
        axis.plot(ground_truth[:, dimension], label="ground truth", linewidth=2)
        axis.plot(predicted[:, dimension], label="GR00T prediction", alpha=0.85)
        axis.set_ylabel(f"action[{dimension}]")
        axis.grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("frame")
    figure.suptitle("Held-out Isaac Lab trajectory: action prediction evidence")
    figure.savefig(output_dir / "heldout_action_trace.png", dpi=180)
    plt.close(figure)

    frames = np.linspace(0, len(sample_arrays["ground_truth"]) - 1, 4, dtype=int)
    figure, axes = plt.subplots(1, len(frames), figsize=(16, 4.5), constrained_layout=True)
    frame_error = np.sqrt(
        np.square(sample_arrays["ground_truth"] - sample_arrays["predicted"]).mean(axis=1)
    )
    for axis, frame_index in zip(axes, frames):
        axis.imshow(sample_trajectory[f"video.{video_key}"].iloc[frame_index])
        axis.set_title(f"frame {frame_index}\naction RMSE={frame_error[frame_index]:.3f}")
        axis.axis("off")
    figure.suptitle("Held-out Isaac Lab G1 demonstration and GR00T action error")
    figure.savefig(output_dir / "isaaclab_heldout_evidence.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-ids", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--execution-horizon", type=int, default=40)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260913)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from humanoidarena_gr00t_video import install_packed_video_offset_patch

    install_packed_video_offset_patch()
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=str(args.model_path.resolve()),
        device="cuda:0",
    )
    policy.model.action_head.num_inference_timesteps = args.denoising_steps
    loader = LeRobotEpisodeLoader(args.dataset_path, policy.get_modality_config())
    torch.cuda.reset_peak_memory_stats()
    rows = []
    sample_arrays = None
    for condition in args.conditions:
        for trajectory_id in args.trajectory_ids:
            row, arrays = evaluate_trajectory(
                policy,
                loader,
                trajectory_id,
                condition,
                execution_horizon=args.execution_horizon,
                steps=args.steps,
                seed=args.seed + trajectory_id * 10_000,
            )
            rows.append(row)
            if sample_arrays is None and condition == "clean":
                sample_arrays = arrays

    summary = {
        "schema_version": 1,
        "model_path": str(args.model_path.resolve()),
        "dataset_path": str(args.dataset_path.resolve()),
        "trajectory_ids": args.trajectory_ids,
        "conditions": args.conditions,
        "execution_horizon": args.execution_horizon,
        "denoising_steps": args.denoising_steps,
        "seed": args.seed,
        "hardware": {
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0),
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        },
        "aggregates": {
            condition: {
                "trajectories": len(selected),
                "mse": statistics.fmean(row["mse"] for row in selected),
                "mae": statistics.fmean(row["mae"] for row in selected),
                "within_0_1_rate": statistics.fmean(row["within_0_1_rate"] for row in selected),
                "mean_inference_ms": statistics.fmean(row["mean_inference_ms"] for row in selected),
            }
            for condition in args.conditions
            if (selected := [row for row in rows if row["condition"] == condition])
        },
        "rows": rows,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "trajectory_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "trajectory_id",
                "condition",
                "frames",
                "mse",
                "mae",
                "rmse",
                "p95_absolute_error",
                "within_0_1_rate",
                "mean_inference_ms",
                "p95_inference_ms",
            ],
        )
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    if sample_arrays is not None:
        np.savez_compressed(args.output_dir / "heldout_predictions.npz", **sample_arrays)
        sample_trajectory = loader[args.trajectory_ids[0]]
        video_keys = loader.modality_configs["video"].modality_keys
        if len(video_keys) != 1:
            raise ValueError(f"evaluation expects exactly one video modality, got {video_keys}")
        video_key = video_keys[0]
        _write_charts(
            rows,
            sample_arrays,
            sample_trajectory,
            args.output_dir,
            video_key,
        )
        source_video = (
            args.dataset_path
            / f"videos/chunk-000/observation.images.{video_key}"
            / f"episode_{args.trajectory_ids[0]:06d}.mp4"
        )
        shutil.copy2(source_video, args.output_dir / "isaaclab_heldout_demonstration.mp4")
    print(json.dumps(summary["aggregates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
