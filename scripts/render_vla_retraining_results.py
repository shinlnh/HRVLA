#!/usr/bin/env python3
"""Render paired VLA post-training metrics and hardware diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np


def paired_statistics(base_rows: list[dict], tuned_rows: list[dict], seed: int = 42) -> dict:
    base = {(row["condition"], row["trajectory_id"]): row for row in base_rows}
    tuned = {(row["condition"], row["trajectory_id"]): row for row in tuned_rows}
    output = {}
    for condition in sorted({key[0] for key in base} & {key[0] for key in tuned}):
        keys = sorted(key for key in base.keys() & tuned.keys() if key[0] == condition)
        base_mse = np.asarray([base[key]["mse"] for key in keys], dtype=np.float64)
        tuned_mse = np.asarray([tuned[key]["mse"] for key in keys], dtype=np.float64)
        delta = base_mse - tuned_mse
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(delta), size=(20_000, len(delta)))
        bootstrap = delta[indices].mean(axis=1)
        positive = int((delta > 0).sum())
        ties = int((delta == 0).sum())
        trials = len(delta) - ties
        tail = min(positive, trials - positive)
        sign_p = min(1.0, 2.0 * sum(math.comb(trials, k) for k in range(tail + 1)) / (2**trials))
        output[condition] = {
            "trajectories": len(keys),
            "base_mse": float(base_mse.mean()),
            "tuned_mse": float(tuned_mse.mean()),
            "mse_reduction": float(delta.mean()),
            "mse_reduction_percent": float(100 * delta.mean() / base_mse.mean()),
            "bootstrap_95_ci": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "improved_trajectories": positive,
            "sign_test_two_sided_p": sign_p,
            "paired_effect_dz": float(delta.mean() / delta.std(ddof=1)) if len(delta) > 1 else 0.0,
        }
    return output


def phase_statistics(base_rows: list[dict], tuned_rows: list[dict]) -> dict:
    """Aggregate clean-condition phase metrics with frame weighting."""
    output = {}
    for label, rows in (("base", base_rows), ("tuned", tuned_rows)):
        accumulators = {}
        for row in rows:
            if row["condition"] != "clean":
                continue
            for phase, metrics in row["phase_metrics"].items():
                values = accumulators.setdefault(phase, {"frames": 0, "mse_sum": 0.0})
                values["frames"] += metrics["frames"]
                values["mse_sum"] += metrics["mse"] * metrics["frames"]
        for phase, values in accumulators.items():
            output.setdefault(phase, {})[f"{label}_mse"] = (
                values["mse_sum"] / values["frames"]
            )
            output[phase]["frames"] = values["frames"]
    for metrics in output.values():
        metrics["mse_reduction_percent"] = 100 * (
            metrics["base_mse"] - metrics["tuned_mse"]
        ) / metrics["base_mse"]
    return output


def parse_gpu_log(path: Path) -> dict:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        try:
            rows.append(
                {
                    "power_w": float(parts[3]),
                    "sm_percent": float(parts[6]),
                    "memory_percent": float(parts[7]),
                    "framebuffer_mib": float(parts[16]),
                }
            )
        except (IndexError, ValueError):
            continue
    if not rows:
        return {}
    result = {"samples": len(rows)}
    for key in rows[0]:
        values = [row[key] for row in rows]
        result[f"{key}_mean"] = statistics.fmean(values)
        result[f"{key}_p95"] = float(np.quantile(values, 0.95))
        result[f"{key}_max"] = max(values)
    return result


def parse_cpu_log(path: Path) -> dict:
    busy = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if "all" not in parts:
            continue
        try:
            busy.append(100.0 - float(parts[-1]))
        except ValueError:
            continue
    return {
        "samples": len(busy),
        "aggregate_busy_percent_mean": statistics.fmean(busy) if busy else None,
        "aggregate_busy_percent_p95": float(np.quantile(busy, 0.95)) if busy else None,
        "aggregate_busy_percent_max": max(busy) if busy else None,
    }


def render_charts(
    paired: dict,
    training_rows: list[dict],
    gpu_log: Path,
    checkpoint_rows: list[dict],
    phase_rows: dict,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = list(paired)
    x = np.arange(len(conditions))
    width = 0.38
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    base = [paired[c]["base_mse"] for c in conditions]
    tuned = [paired[c]["tuned_mse"] for c in conditions]
    axes[0].bar(x - width / 2, base, width, label="base N1.7", color="#718096")
    axes[0].bar(x + width / 2, tuned, width, label="post-trained", color="#2b6cb0")
    axes[0].set_xticks(x, [condition.replace("_", "\n") for condition in conditions])
    axes[0].set_ylabel("held-out action MSE")
    axes[0].set_title("Paired robustness evaluation")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    reductions = [paired[c]["mse_reduction_percent"] for c in conditions]
    bars = axes[1].bar([c.replace("_", "\n") for c in conditions], reductions, color="#2f855a")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("MSE reduction (%)")
    axes[1].set_title("Post-training improvement")
    axes[1].bar_label(bars, fmt="%.2f%%")
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "vla_retraining_comparison.png", dpi=180)
    plt.close(figure)

    steps = np.asarray([row["step"] for row in training_rows if "loss" in row])
    losses = np.asarray([row["loss"] for row in training_rows if "loss" in row])
    window = min(25, len(losses))
    moving = np.convolve(losses, np.ones(window) / window, mode="valid")
    gpu_rows = []
    for line in gpu_log.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        try:
            parts = line.split()
            gpu_rows.append((float(parts[6]), float(parts[16])))
        except (IndexError, ValueError):
            pass
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].plot(steps, losses, alpha=0.25, label="step loss")
    axes[0].plot(steps[window - 1 :], moving, linewidth=2, label=f"{window}-step mean")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("flow-matching loss")
    axes[0].set_title("Training convergence")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    if gpu_rows:
        gpu = np.asarray(gpu_rows)
        axes[1].plot(gpu[:, 0], label="SM utilization (%)")
        axes[1].plot(gpu[:, 1] / 16303 * 100, label="VRAM used (%)")
    axes[1].set_xlabel("one-second sample")
    axes[1].set_ylabel("percent")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("RTX 5070 Ti utilization")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.savefig(output_dir / "training_diagnostics.png", dpi=180)
    plt.close(figure)

    if checkpoint_rows:
        figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
        checkpoint_rows = sorted(checkpoint_rows, key=lambda row: row["step"])
        bars = axis.plot(
            [row["step"] for row in checkpoint_rows],
            [row["clean_mse"] for row in checkpoint_rows],
            marker="o",
            linewidth=2,
            color="#2b6cb0",
        )
        del bars
        for row in checkpoint_rows:
            axis.annotate(
                f'{row["clean_mse"]:.6f}',
                (row["step"], row["clean_mse"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
            )
        axis.set_xlabel("optimizer step (0 = calibrated base)")
        axis.set_ylabel("held-out clean action MSE")
        axis.set_title("Checkpoint trend (reporting only; no test-set selection)")
        axis.grid(alpha=0.25)
        figure.savefig(output_dir / "checkpoint_trend.png", dpi=180)
        plt.close(figure)

    if phase_rows:
        labels = list(phase_rows)
        short_labels = [f"phase {index + 1}" for index in range(len(labels))]
        x = np.arange(len(labels))
        figure, axis = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
        axis.bar(
            x - width / 2,
            [phase_rows[label]["base_mse"] for label in labels],
            width,
            label="base N1.7",
            color="#718096",
        )
        axis.bar(
            x + width / 2,
            [phase_rows[label]["tuned_mse"] for label in labels],
            width,
            label="post-trained",
            color="#2b6cb0",
        )
        axis.set_xticks(x, short_labels)
        axis.set_ylabel("held-out clean action MSE")
        axis.set_title("Per-subtask result (phase labels are listed in summary.json)")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        figure.savefig(output_dir / "subtask_phase_comparison.png", dpi=180)
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-metrics", type=Path, required=True)
    parser.add_argument("--tuned-metrics", type=Path, required=True)
    parser.add_argument("--trainer-state", type=Path, required=True)
    parser.add_argument("--gpu-log", type=Path, required=True)
    parser.add_argument("--cpu-log", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-metrics",
        type=Path,
        nargs="*",
        default=[],
        help="Optional checkpoint metrics.json files used for the reporting-only trend",
    )
    parser.add_argument("--physical-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = json.loads(args.base_metrics.read_text(encoding="utf-8"))
    tuned = json.loads(args.tuned_metrics.read_text(encoding="utf-8"))
    trainer = json.loads(args.trainer_state.read_text(encoding="utf-8"))
    paired = paired_statistics(base["rows"], tuned["rows"])
    phases = phase_statistics(base["rows"], tuned["rows"])
    losses = [row["loss"] for row in trainer["log_history"] if "loss" in row]
    checkpoint_rows = [{"step": 0, "clean_mse": base["aggregates"]["clean"]["mse"]}]
    for path in args.checkpoint_metrics:
        match = re.search(r"checkpoint-(\d+)", str(path))
        if not match:
            raise ValueError(f"cannot derive checkpoint step from {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        checkpoint_rows.append(
            {
                "step": int(match.group(1)),
                "clean_mse": metrics["aggregates"]["clean"]["mse"],
            }
        )
    checkpoint_rows.append(
        {
            "step": trainer["global_step"],
            "clean_mse": tuned["aggregates"]["clean"]["mse"],
        }
    )
    summary = {
        "schema_version": 1,
        "paired_evaluation": paired,
        "phase_evaluation": phases,
        "training": {
            "optimizer_steps": trainer["global_step"],
            "physical_batch_size": args.physical_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": (
                args.physical_batch_size * args.gradient_accumulation_steps
            ),
            "first_25_loss_mean": statistics.fmean(losses[:25]),
            "last_25_loss_mean": statistics.fmean(losses[-25:]),
            "loss_change_percent": 100
            * (
                statistics.fmean(losses[-25:])
                / statistics.fmean(losses[:25])
                - 1
            ),
            "checkpoint_clean_mse": sorted(
                checkpoint_rows, key=lambda row: row["step"]
            ),
        },
        "hardware": {
            "gpu": parse_gpu_log(args.gpu_log),
            "cpu": parse_cpu_log(args.cpu_log),
        },
        "claim_boundary": (
            "Open-loop action prediction on held-out Isaac Lab demonstrations; "
            "not a closed-loop manipulation success rate."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "paired_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "condition",
                "trajectories",
                "base_mse",
                "tuned_mse",
                "mse_reduction",
                "mse_reduction_percent",
                "improved_trajectories",
                "sign_test_two_sided_p",
                "paired_effect_dz",
            ],
        )
        writer.writeheader()
        for condition, metrics in paired.items():
            writer.writerow(
                {
                    "condition": condition,
                    **{key: metrics[key] for key in writer.fieldnames[1:]},
                }
            )
    render_charts(
        paired,
        trainer["log_history"],
        args.gpu_log,
        checkpoint_rows,
        phases,
        args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
