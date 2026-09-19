#!/usr/bin/env python3
"""Freeze and render the paired common-vs-ST-RT validation evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.humanoidarena_rt_training import (  # noqa: E402
    load_rt_lock,
    select_checkpoint,
)
from hrvla_bench.humanoidarena_training import (  # noqa: E402
    load_training_lock,
    validate_selection,
)
from hrvla_bench.plan import canonical_sha256  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _seed_macros(selection: dict[str, Any], step: int) -> dict[str, float]:
    return {
        seed: float(row["macro_task_mse"])
        for seed, row in selection["candidates"][str(step)]["seeds"].items()
    }


def build_report(
    common: dict[str, Any], st_rt: dict[str, Any], *, common_manifest: str, st_manifest: str
) -> dict[str, Any]:
    steps = (100, 200, 300)
    if common.get("hidden_test_accessed") is not False or st_rt.get("hidden_test_accessed") is not False:
        raise ValueError("hidden data must remain inaccessible during validation comparison")
    if set(common["candidates"]) != {str(step) for step in steps} or set(
        st_rt["candidates"]
    ) != {str(step) for step in steps}:
        raise ValueError("validation candidates must be exactly steps 100/200/300")
    rows = []
    for step in steps:
        common_seeds = _seed_macros(common, step)
        st_seeds = _seed_macros(st_rt, step)
        if set(common_seeds) != {"0", "1", "2"} or set(st_seeds) != set(common_seeds):
            raise ValueError("validation comparison requires the same three seeds")
        paired = {
            seed: {
                "common_macro_task_mse": common_seeds[seed],
                "st_rt_macro_task_mse": st_seeds[seed],
                "absolute_reduction": common_seeds[seed] - st_seeds[seed],
                "reduction_percent": 100.0
                * (common_seeds[seed] - st_seeds[seed])
                / common_seeds[seed],
            }
            for seed in sorted(common_seeds)
        }
        rows.append(
            {
                "step": step,
                "common_mean_seed_macro_task_mse": statistics.fmean(common_seeds.values()),
                "st_rt_mean_seed_macro_task_mse": statistics.fmean(st_seeds.values()),
                "paired_by_seed": paired,
                "mean_paired_reduction_percent": statistics.fmean(
                    row["reduction_percent"] for row in paired.values()
                ),
            }
        )
    selected_step = int(st_rt["selected_step"])
    selected = next(row for row in rows if row["step"] == selected_step)
    core = {
        "schema_version": 1,
        "claim_boundary": (
            "validation-only open-loop action error; this is neither held-out evidence "
            "nor a closed-loop task-success or recovery-success claim"
        ),
        "hidden_test_accessed": False,
        "common_selection_sha256": common["selection_sha256"],
        "st_rt_selection_sha256": st_rt["selection_sha256"],
        "common_validation_manifest_sha256": common_manifest,
        "st_rt_validation_manifest_sha256": st_manifest,
        "common_selected_step": int(common["selected_step"]),
        "st_rt_selected_step": selected_step,
        "candidates": rows,
        "selected_comparison": selected,
    }
    return {**core, "audit_sha256": canonical_sha256(core)}


def _render(report: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["candidates"]
    steps = [row["step"] for row in rows]
    figure, (curve, paired_axis) = plt.subplots(
        1, 2, figsize=(13.5, 5.2), constrained_layout=True
    )
    curve.plot(
        steps,
        [row["common_mean_seed_macro_task_mse"] for row in rows],
        marker="o",
        label="Common checkpoint",
    )
    curve.plot(
        steps,
        [row["st_rt_mean_seed_macro_task_mse"] for row in rows],
        marker="o",
        label="ST-RT",
    )
    curve.axvline(report["st_rt_selected_step"], color="#2f855a", linestyle="--")
    curve.set_xticks(steps)
    curve.set_xlabel("Optimizer step")
    curve.set_ylabel("Mean seed macro-per-task action MSE")
    curve.set_title("Validation checkpoint selection")
    curve.grid(alpha=0.25)
    curve.legend()

    selected = report["selected_comparison"]
    paired = selected["paired_by_seed"]
    seeds = sorted(paired)
    reductions = [paired[seed]["reduction_percent"] for seed in seeds]
    bars = paired_axis.bar([f"seed {seed}" for seed in seeds], reductions, color="#2b7bba")
    paired_axis.bar_label(bars, fmt="%.3f%%")
    paired_axis.axhline(0, color="black", linewidth=0.8)
    paired_axis.set_ylabel("Common → ST-RT MSE reduction")
    paired_axis.set_title(
        f"Paired effect at selected step {report['st_rt_selected_step']}"
    )
    paired_axis.grid(axis="y", alpha=0.25)
    figure.suptitle("HumanoidArena ST-RT validation — open-loop evidence only")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--common-lock",
        type=Path,
        default=ROOT / "config/humanoidarena-gr00t-training.lock.json",
    )
    parser.add_argument(
        "--rt-lock", type=Path, default=ROOT / "config/humanoidarena-rt-training.lock.json"
    )
    parser.add_argument(
        "--common-selection",
        type=Path,
        default=ROOT / "_artifacts/retraining/humanoidarena-common/selection.json",
    )
    parser.add_argument(
        "--st-selection",
        type=Path,
        default=ROOT / "_artifacts/retraining/humanoidarena-rt/st-rt/selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/benchmark/retraining/humanoidarena_st_rt_validation.json",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=ROOT / "results/benchmark/retraining/humanoidarena_st_rt_validation.png",
    )
    args = parser.parse_args()

    common_lock = load_training_lock(args.common_lock.resolve())
    rt_lock = load_rt_lock(args.rt_lock.resolve())
    common = _load(args.common_selection.resolve())
    st_rt = _load(args.st_selection.resolve())
    validate_selection(common, common_lock)
    expected_st = select_checkpoint(ROOT, rt_lock, "gr00t_st_rt")
    if expected_st != st_rt:
        raise ValueError("ST-RT selection differs from the complete validation evidence")
    report = build_report(
        common,
        st_rt,
        common_manifest=common["dataset_manifest_sha256"],
        st_manifest=rt_lock["subtask_rt_dataset"]["manifests"]["validation"],
    )
    _write_json(args.output.resolve(), report)
    _render(report, args.plot.resolve())
    print(json.dumps({"audit_sha256": report["audit_sha256"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
