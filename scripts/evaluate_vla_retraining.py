#!/usr/bin/env python3
"""Paired open-loop evaluation for in-domain GR00T VLA checkpoints."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import shutil
import statistics
import sys
import threading
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.evaluation_batching import (
    bounded_ordered_prefetch,
    inject_seeded_action_noise,
    stack_observations,
)


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


@dataclass(frozen=True)
class PreparedRequest:
    condition: str
    step: int
    seed: int
    observation: dict[str, Any]


@dataclass(frozen=True)
class PreparedTrajectory:
    trajectory_id: int
    frame: Any
    actual_steps: int
    ground_truth: np.ndarray
    phase_text: np.ndarray
    requests: tuple[PreparedRequest, ...]


def _prepare_trajectory(
    loader,
    trajectory_id: int,
    conditions: list[str],
    *,
    embodiment_tag,
    language_key: str,
    execution_horizon: int,
    steps: int,
    seed: int,
) -> PreparedTrajectory:
    """Decode one trajectory once and materialize every deterministic condition."""

    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.utils import parse_observation_gr00t

    trajectory = loader[trajectory_id]
    actual_steps = min(steps, len(trajectory))
    input_config = deepcopy(loader.modality_configs)
    input_config.pop("action")
    requests = []
    for condition in conditions:
        for step in range(0, actual_steps, execution_horizon):
            request_seed = seed + step
            point = extract_step_data(trajectory, step, input_config, embodiment_tag)
            flat_observation = {
                **{f"video.{key}": np.asarray(value) for key, value in point.images.items()},
                **{
                    f"state.{key}": np.asarray(value, dtype=np.float32)
                    for key, value in point.states.items()
                },
                language_key: point.text,
            }
            parsed = parse_observation_gr00t(flat_observation, loader.modality_configs)
            _corrupt_observation(parsed, condition, np.random.default_rng(request_seed))
            requests.append(
                PreparedRequest(
                    condition=condition,
                    step=step,
                    seed=request_seed,
                    observation=parsed,
                )
            )
    action_keys = loader.modality_configs["action"].modality_keys
    ground_truth = _concatenate_columns(
        trajectory, [f"action.{key}" for key in action_keys]
    )[:actual_steps]
    phase_text = np.asarray(trajectory[f"language.{language_key}"].tolist())[:actual_steps]
    return PreparedTrajectory(
        trajectory_id=trajectory_id,
        frame=trajectory,
        actual_steps=actual_steps,
        ground_truth=ground_truth,
        phase_text=phase_text,
        requests=tuple(requests),
    )


def _trajectory_metrics(
    prepared: PreparedTrajectory,
    condition: str,
    predicted: np.ndarray,
    inference_latencies: list[float],
    *,
    timing_semantics: str,
) -> tuple[dict, dict[str, np.ndarray]]:
    ground_truth = prepared.ground_truth
    absolute_error = np.abs(ground_truth - predicted)
    squared_error = np.square(ground_truth - predicted)
    phase_metrics = {}
    for phase in sorted(set(prepared.phase_text.tolist())):
        mask = prepared.phase_text == phase
        phase_metrics[phase] = {
            "frames": int(mask.sum()),
            "mse": float(squared_error[mask].mean()),
            "mae": float(absolute_error[mask].mean()),
        }
    metrics = {
        "trajectory_id": prepared.trajectory_id,
        "condition": condition,
        "frames": prepared.actual_steps,
        "mse": float(squared_error.mean()),
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(squared_error.mean())),
        "p95_absolute_error": float(np.quantile(absolute_error, 0.95)),
        "within_0_1_rate": float((absolute_error < 0.1).mean()),
        "mean_inference_ms": statistics.fmean(inference_latencies),
        "p95_inference_ms": float(np.quantile(inference_latencies, 0.95)),
        "inference_timing_semantics": timing_semantics,
        "phase_metrics": phase_metrics,
    }
    return metrics, {
        "ground_truth": ground_truth,
        "predicted": predicted,
        "phase_text": prepared.phase_text,
    }


def evaluate_prepared_trajectory(
    policy,
    loader,
    prepared: PreparedTrajectory,
    conditions: list[str],
    *,
    batch_size: int,
    execution_horizon: int,
) -> list[tuple[dict, dict[str, np.ndarray]]]:
    """Consume CPU-prepared requests in deterministic GPU batches."""

    action_keys = loader.modality_configs["action"].modality_keys
    chunks: dict[str, list[tuple[int, np.ndarray]]] = {name: [] for name in conditions}
    latencies: dict[str, list[float]] = {name: [] for name in conditions}
    requests = list(prepared.requests)
    for start in range(0, len(requests), batch_size):
        batch = requests[start : start + batch_size]
        observation = stack_observations([request.observation for request in batch])
        torch.cuda.synchronize()
        started = time.perf_counter()
        with inject_seeded_action_noise(policy, [request.seed for request in batch]):
            action, _ = policy.get_action(observation)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1_000
        amortized_ms = elapsed_ms / len(batch)
        for batch_index, request in enumerate(batch):
            action_chunk = np.concatenate(
                [np.asarray(action[key])[batch_index] for key in action_keys], axis=-1
            )
            remaining = prepared.actual_steps - request.step
            chunks[request.condition].append(
                (request.step, action_chunk[: min(execution_horizon, remaining)])
            )
            latencies[request.condition].append(amortized_ms)

    output = []
    for condition in conditions:
        predicted = np.concatenate(
            [value for _, value in sorted(chunks[condition], key=lambda item: item[0])], axis=0
        )[: prepared.actual_steps]
        output.append(
            _trajectory_metrics(
                prepared,
                condition,
                predicted,
                latencies[condition],
                timing_semantics="amortized_gpu_batch_wall_time_non_claim",
            )
        )
    return output


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
    parser.add_argument(
        "--throughput-batch-size",
        type=int,
        default=1,
        help="batch independent observations on GPU; 1 retains the serial reference path",
    )
    parser.add_argument(
        "--prefetch-workers",
        type=int,
        default=1,
        help="bounded CPU trajectory preparation workers for the batched path",
    )
    args = parser.parse_args()
    if args.throughput_batch_size < 1:
        parser.error("--throughput-batch-size must be positive")
    if args.prefetch_workers < 1:
        parser.error("--prefetch-workers must be positive")
    if args.throughput_batch_size == 1 and args.prefetch_workers != 1:
        parser.error("--prefetch-workers requires --throughput-batch-size greater than one")
    return args


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
    sample_trajectory = None
    evaluation_started = time.perf_counter()
    if args.throughput_batch_size == 1:
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
                row["inference_timing_semantics"] = "serial_request_wall_time"
                rows.append(row)
                if sample_arrays is None and condition == "clean":
                    sample_arrays = arrays
    else:
        worker_state = threading.local()
        modality_configs = deepcopy(policy.get_modality_config())

        def prepare(trajectory_id: int) -> PreparedTrajectory:
            if not hasattr(worker_state, "loader"):
                worker_state.loader = LeRobotEpisodeLoader(
                    args.dataset_path, deepcopy(modality_configs)
                )
            return _prepare_trajectory(
                worker_state.loader,
                trajectory_id,
                args.conditions,
                embodiment_tag=policy.embodiment_tag,
                language_key=policy.language_key,
                execution_horizon=args.execution_horizon,
                steps=args.steps,
                seed=args.seed + trajectory_id * 10_000,
            )

        for prepared in bounded_ordered_prefetch(
            args.trajectory_ids,
            prepare,
            workers=args.prefetch_workers,
        ):
            evaluated = evaluate_prepared_trajectory(
                policy,
                loader,
                prepared,
                args.conditions,
                batch_size=args.throughput_batch_size,
                execution_horizon=args.execution_horizon,
            )
            for row, arrays in evaluated:
                rows.append(row)
                if sample_arrays is None and row["condition"] == "clean":
                    sample_arrays = arrays
                    sample_trajectory = prepared.frame
    evaluation_wall_seconds = time.perf_counter() - evaluation_started

    summary = {
        "schema_version": 1,
        "model_path": str(args.model_path.resolve()),
        "dataset_path": str(args.dataset_path.resolve()),
        "trajectory_ids": args.trajectory_ids,
        "conditions": args.conditions,
        "execution_horizon": args.execution_horizon,
        "denoising_steps": args.denoising_steps,
        "seed": args.seed,
        "execution": {
            "profile": (
                "serial_reference"
                if args.throughput_batch_size == 1
                else "cpu_prefetch_gpu_batch"
            ),
            "throughput_batch_size": args.throughput_batch_size,
            "prefetch_workers": args.prefetch_workers,
            "wall_seconds": evaluation_wall_seconds,
            "latency_claim_eligible": args.throughput_batch_size == 1,
            "batched_timing_semantics": (
                None
                if args.throughput_batch_size == 1
                else "amortized_gpu_batch_wall_time_non_claim"
            ),
        },
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
        if sample_trajectory is None:
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
