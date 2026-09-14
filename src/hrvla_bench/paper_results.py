"""Aggregate locked multi-seed VLA evaluation results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np


CONDITIONS = ("clean", "vision_noise", "occlusion", "state_noise", "combined")
METHOD_RUN_NAMES = {"ST-RT": "st", "STR-RT": "str"}


def load_metrics(path: Path) -> dict[str, Any]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    conditions = tuple(metrics.get("conditions", ()))
    trajectory_ids = metrics.get("trajectory_ids", [])
    if conditions != CONDITIONS:
        raise ValueError(f"unexpected evaluation conditions in {path}")
    if trajectory_ids != list(range(42)):
        raise ValueError(f"expected trajectory IDs 0..41 in {path}")
    if len(metrics.get("rows", [])) != len(CONDITIONS) * 42:
        raise ValueError(f"expected 210 evaluation rows in {path}")
    return metrics


def _hierarchical_bootstrap(
    deltas_by_seed: list[np.ndarray], *, samples: int = 20_000, seed: int = 20260913
) -> list[float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    seed_count = len(deltas_by_seed)
    for index in range(samples):
        selected_seeds = rng.integers(0, seed_count, size=seed_count)
        seed_means = []
        for selected_seed in selected_seeds:
            values = deltas_by_seed[selected_seed]
            selected_rows = rng.integers(0, len(values), size=len(values))
            seed_means.append(float(values[selected_rows].mean()))
        estimates[index] = statistics.fmean(seed_means)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def summarize(evaluation_root: Path, seeds: list[int]) -> dict[str, Any]:
    loaded: dict[str, list[dict[str, Any]]] = {}
    for method, run_name in METHOD_RUN_NAMES.items():
        loaded[method] = [
            load_metrics(evaluation_root / f"{run_name}-seed-{seed}/metrics.json")
            for seed in seeds
        ]

    methods: dict[str, Any] = {}
    for method, runs in loaded.items():
        methods[method] = {}
        for condition in CONDITIONS:
            seed_mse = [run["aggregates"][condition]["mse"] for run in runs]
            methods[method][condition] = {
                "seed_mse": seed_mse,
                "mean_mse": statistics.fmean(seed_mse),
                "sample_std_mse": statistics.stdev(seed_mse),
            }

    paired: dict[str, Any] = {}
    for condition in CONDITIONS:
        deltas_by_seed = []
        for st_run, str_run in zip(loaded["ST-RT"], loaded["STR-RT"], strict=True):
            st_rows = {
                row["trajectory_id"]: row
                for row in st_run["rows"]
                if row["condition"] == condition
            }
            str_rows = {
                row["trajectory_id"]: row
                for row in str_run["rows"]
                if row["condition"] == condition
            }
            if st_rows.keys() != str_rows.keys():
                raise ValueError(f"unpaired trajectory rows for {condition}")
            deltas_by_seed.append(
                np.asarray(
                    [st_rows[key]["mse"] - str_rows[key]["mse"] for key in sorted(st_rows)],
                    dtype=np.float64,
                )
            )
        seed_reductions = [float(values.mean()) for values in deltas_by_seed]
        st_mean = methods["ST-RT"][condition]["mean_mse"]
        mean_reduction = statistics.fmean(seed_reductions)
        paired[condition] = {
            "seed_mse_reductions": seed_reductions,
            "mean_mse_reduction": mean_reduction,
            "sample_std_mse_reduction": statistics.stdev(seed_reductions),
            "mean_reduction_percent": 100 * mean_reduction / st_mean,
            "hierarchical_bootstrap_95_ci": _hierarchical_bootstrap(deltas_by_seed),
            "improved_seed_count": sum(value > 0 for value in seed_reductions),
            "seed_count": len(seeds),
            "trajectories_per_seed": 42,
        }

    return {
        "schema_version": 1,
        "protocol": {
            "training_seeds": seeds,
            "evaluation_seed": 20260913,
            "conditions": list(CONDITIONS),
            "trajectories_per_condition": 42,
            "steps_per_trajectory": 120,
            "denoising_steps": 4,
            "bootstrap_resamples": 20_000,
            "bootstrap_unit": "training seed, then trajectory within seed",
        },
        "methods": methods,
        "paired_str_rt_minus_st_rt": paired,
    }


def write_summary(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "condition_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "condition",
            "st_rt_mean_mse",
            "st_rt_sample_std_mse",
            "str_rt_mean_mse",
            "str_rt_sample_std_mse",
            "mean_mse_reduction",
            "mean_reduction_percent",
            "bootstrap_95_ci_low",
            "bootstrap_95_ci_high",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for condition in CONDITIONS:
            st = summary["methods"]["ST-RT"][condition]
            recovery = summary["methods"]["STR-RT"][condition]
            paired = summary["paired_str_rt_minus_st_rt"][condition]
            writer.writerow(
                {
                    "condition": condition,
                    "st_rt_mean_mse": st["mean_mse"],
                    "st_rt_sample_std_mse": st["sample_std_mse"],
                    "str_rt_mean_mse": recovery["mean_mse"],
                    "str_rt_sample_std_mse": recovery["sample_std_mse"],
                    "mean_mse_reduction": paired["mean_mse_reduction"],
                    "mean_reduction_percent": paired["mean_reduction_percent"],
                    "bootstrap_95_ci_low": paired["hierarchical_bootstrap_95_ci"][0],
                    "bootstrap_95_ci_high": paired["hierarchical_bootstrap_95_ci"][1],
                }
            )
