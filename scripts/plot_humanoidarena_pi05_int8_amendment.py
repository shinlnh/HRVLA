#!/usr/bin/env python3
"""Plot the provenance-explicit PI0.5 INT8 external matrix summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/benchmark/external/pi05_cuda_int8_v1_amended_summary.json"
DEFAULT_OUTPUT = ROOT / "results/benchmark/external/pi05_cuda_int8_v1_amended_summary.png"
TASKS = ("boxing", "doubledesk", "football", "open_door", "pp_box", "sit_sofa", "vision_navi")
MODES = ("base_test", "semantic", "vision", "execution")


def plot_summary(summary: dict, output: Path) -> None:
    if (
        summary.get("claim_status") != "amended_int8_matrix_complete_cpu_equivalence_not_proven"
        or summary.get("matrix", {}).get("episodes") != 1680
        or summary.get("amendment", {}).get("selected_corrected_episodes") != 240
    ):
        raise ValueError("input is not the validated amended INT8 matrix")

    by_task = summary["by_task"]
    by_task_mode = summary["by_task_mode"]
    rates = np.array([by_task[task]["success_rate"] for task in TASKS])
    intervals = np.array([by_task[task]["success_rate_wilson_95"] for task in TASKS])
    heatmap = np.array([
        [by_task_mode[f"{mode}/{task}"]["success_rate"] for mode in MODES]
        for task in TASKS
    ])

    fig, (ax, heat_ax) = plt.subplots(
        1, 2, figsize=(14, 6.4), gridspec_kw={"width_ratios": [1, 1.15]}
    )
    y = np.arange(len(TASKS))
    colors = ["#65748b" if task == "open_door" else "#24759b" for task in TASKS]
    colors[TASKS.index("pp_box")] = "#d28631"
    ax.barh(y, rates * 100, color=colors, height=0.65)
    ax.errorbar(
        rates * 100,
        y,
        xerr=np.vstack(((rates - intervals[:, 0]) * 100, (intervals[:, 1] - rates) * 100)),
        fmt="none", ecolor="#233041", capsize=3, linewidth=1.2,
    )
    ax.set_yticks(y, ["open_door (diagnostic)" if task == "open_door" else task for task in TASKS])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Success rate (%) with Wilson 95% CI")
    ax.set_title("Per-task closed-loop success (n = 240 each)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for index, interval in enumerate(intervals):
        ax.text(min(interval[1] * 100 + 2, 97), index,
                f"{by_task[TASKS[index]]['successes']}/240", va="center", fontsize=8)

    image = heat_ax.imshow(heatmap * 100, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    heat_ax.set_xticks(np.arange(len(MODES)), MODES, rotation=25, ha="right")
    heat_ax.set_yticks(y, TASKS)
    heat_ax.set_title("Success rate by task × condition (n = 60/cell)")
    for row in range(len(TASKS)):
        for col in range(len(MODES)):
            value = heatmap[row, col] * 100
            heat_ax.text(col, row, f"{value:.1f}%", ha="center", va="center",
                         color="white" if value > 53 else "#172233", fontsize=8)
    fig.colorbar(image, ax=heat_ax, label="Success rate (%)", fraction=0.046, pad=0.04)

    overall = summary["overall"]
    primary = summary["primary_six_tasks_excluding_open_door"]
    fig.suptitle(
        "PI0.5 + SONIC · CUDA INT8 external matrix (amended PickPlaceBox prompt)",
        fontsize=14, y=0.985,
    )
    fig.text(
        0.5, 0.035,
        f"Overall: {overall['successes']}/{overall['episodes']} ({overall['success_rate']:.1%}); "
        f"six-task primary: {primary['successes']}/{primary['episodes']} ({primary['success_rate']:.1%}). "
        "PickPlaceBox uses corrected rerun; original invalid-prompt rows are excluded. "
        "CPU–INT8 equivalence is not established.",
        ha="center", fontsize=8,
    )
    fig.subplots_adjust(left=0.16, right=0.94, top=0.88, bottom=0.17, wspace=0.34)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, metadata={"Software": "HRVLA PI0.5 INT8 amendment plotter"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot_summary(json.loads(args.input.read_text(encoding="utf-8")), args.output)
    print(f"[pi05-amendment-plot] output={args.output}")


if __name__ == "__main__":
    main()
