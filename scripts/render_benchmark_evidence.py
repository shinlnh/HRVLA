#!/usr/bin/env python3
"""Render compact, claim-bounded plots from committed benchmark evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RETRAINING = ROOT / "results" / "retraining" / "paper-seeds" / "summary.json"
DEFAULT_HUMANOID = ROOT / "results" / "humanoidarena" / "baseline-matrix" / "summary.json"
DEFAULT_OUTPUT = ROOT / "results" / "benchmark" / "readiness"


def readiness_rows(humanoid: dict[str, Any] | None) -> list[dict[str, Any]]:
    matrix = (humanoid or {}).get("matrix", {})
    cells_complete = int(matrix.get("cells_complete", 0))
    cells_expected = int(matrix.get("cells_expected", 84))
    return [
        {
            "workstream": "Planner component",
            "completed": 5,
            "expected": 5,
            "unit": "algorithm variants",
            "scope": "symbolic/Cosmos component evidence",
        },
        {
            "workstream": "Recovery component",
            "completed": 6,
            "expected": 6,
            "unit": "mechanism variants",
            "scope": "symbolic fault-injection evidence",
        },
        {
            "workstream": "VLA retraining",
            "completed": 6,
            "expected": 6,
            "unit": "training seeds",
            "scope": "held-out open-loop evaluation",
        },
        {
            "workstream": "PI0.5 + SONIC",
            "completed": cells_complete,
            "expected": cells_expected,
            "unit": "HumanoidArena cells",
            "scope": "direct external rerun; partial non-claim",
        },
        {
            "workstream": "Scenario admission",
            "completed": 0,
            "expected": 9,
            "unit": "recovery scenarios",
            "scope": "oracle 20/20 plus immutable failure artifacts",
        },
        {
            "workstream": "Internal closed loop",
            "completed": 0,
            "expected": 5,
            "unit": "registered methods",
            "scope": "same frozen paired HumanoidArena plan",
        },
    ]


def _load_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_readiness(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "claim_boundary": "workstreams are not interchangeable; do not average percentages",
        "workstreams": [
            {
                **row,
                "percent": 100.0 * row["completed"] / row["expected"],
            }
            for row in rows
        ],
    }
    (output_dir / "status.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "status.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("workstream", "completed", "expected", "unit", "percent", "scope"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(payload["workstreams"])


def _render_retraining(summary: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    conditions = list(summary["protocol"]["conditions"])
    labels = [condition.replace("_", "\n") for condition in conditions]
    positions = list(range(len(conditions)))
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    width = 0.36
    series = (("ST-RT", -width / 2, "#718096"), ("STR-RT", width / 2, "#2b6cb0"))
    for method, offset, color in series:
        means = [summary["methods"][method][condition]["mean_mse"] for condition in conditions]
        errors = [
            summary["methods"][method][condition]["sample_std_mse"]
            for condition in conditions
        ]
        axes[0].bar(
            [position + offset for position in positions],
            means,
            width,
            yerr=errors,
            capsize=3,
            label=method,
            color=color,
        )
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("held-out action MSE (mean ± seed SD)")
    axes[0].set_title("Three-seed VLA replication")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    reductions = [
        summary["paired_str_rt_minus_st_rt"][condition]["mean_reduction_percent"]
        for condition in conditions
    ]
    bars = axes[1].bar(labels, reductions, color="#2f855a")
    axes[1].bar_label(bars, fmt="%.3f%%")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("paired MSE reduction (%)")
    axes[1].set_title("STR-RT improvement over ST-RT")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Open-loop evidence only — not closed-loop task or recovery success", fontsize=10
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _render_readiness(rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["workstream"] for row in rows]
    percentages = [100.0 * row["completed"] / row["expected"] for row in rows]
    colors = [
        "#2f855a" if value == 100 else "#dd6b20" if value > 0 else "#c53030"
        for value in percentages
    ]
    figure, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    bars = axis.barh(labels[::-1], percentages[::-1], color=colors[::-1])
    axis.set_xlim(0, 108)
    axis.set_xlabel("protocol completion (%)")
    axis.set_title(
        "Benchmark readiness by independent workstream\n"
        "Rows use different units and must not be averaged"
    )
    axis.grid(axis="x", alpha=0.25)
    for bar, row in zip(bars, rows[::-1]):
        axis.text(
            min(bar.get_width() + 1.2, 102),
            bar.get_y() + bar.get_height() / 2,
            f"{row['completed']}/{row['expected']}",
            va="center",
            fontsize=9,
        )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _render_humanoid(summary: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    by_task_mode = summary.get("by_task_mode", {})
    keys = list(by_task_mode)
    labels = [key.replace("base_test/", "base/").replace("_", " ") for key in keys]
    rates = [100.0 * by_task_mode[key]["success_rate"] for key in keys]
    trials = [int(by_task_mode[key]["episodes"]) for key in keys]
    matrix = summary["matrix"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    if keys:
        bars = axes[0].barh(labels[::-1], rates[::-1], color="#2b6cb0")
        for bar, count in zip(bars, trials[::-1]):
            axes[0].text(
                min(bar.get_width() + 1, 96),
                bar.get_y() + bar.get_height() / 2,
                f"n={count}",
                va="center",
                fontsize=9,
            )
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("observed success rate (%)")
    axes[0].set_title("Observed task/mode slices")
    axes[0].grid(axis="x", alpha=0.25)

    observed = int(matrix["episodes_complete"])
    expected = int(matrix["episodes_expected"])
    completion = [
        100 * observed / expected,
        100 * matrix["cells_complete"] / matrix["cells_expected"],
    ]
    axes[1].bar(
        ["episodes", "complete cells"], completion, color=["#dd6b20", "#805ad5"]
    )
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("matrix completion (%)")
    axes[1].set_title(f"PI0.5+SONIC: {observed}/{expected} observed episodes")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Interrupted partial evidence — non-claim until all locked cells pass", fontsize=10
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retraining-summary", type=Path, default=DEFAULT_RETRAINING)
    parser.add_argument("--humanoidarena-summary", type=Path, default=DEFAULT_HUMANOID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    output_dir = args.output_dir.resolve()
    humanoid = _load_optional(args.humanoidarena_summary.resolve())
    retraining = _load_optional(args.retraining_summary.resolve())
    rows = readiness_rows(humanoid)
    _write_readiness(rows, output_dir)
    _render_readiness(rows, output_dir / "benchmark_readiness.png")
    if retraining is not None:
        retraining_output = args.retraining_summary.resolve().parent / "multiseed_comparison.png"
        _render_retraining(retraining, retraining_output)
    if humanoid is not None:
        humanoid_output = args.humanoidarena_summary.resolve().parent / "partial_progress.png"
        _render_humanoid(humanoid, humanoid_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
