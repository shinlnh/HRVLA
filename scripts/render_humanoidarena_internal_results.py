#!/usr/bin/env python3
"""Audit, score, and render the claim-bearing internal hidden-final matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.internal_protocol import INTERNAL_METHODS  # noqa: E402
from hrvla_bench.internal_report import build_internal_report  # noqa: E402
from hrvla_bench.plan import load_json  # noqa: E402
from hrvla_bench.score import load_jsonl  # noqa: E402


def _load_jsonl_unvalidated(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _success_plot(report: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(report["methods"])
    protocols = ("nominal", "failure_start", "online_failure")
    labels = {
        "nominal": "Nominal SR",
        "failure_start": "Failure-start RSR",
        "online_failure": "Online RSR",
    }
    colors = {"nominal": "#2b6cb0", "failure_start": "#805ad5", "online_failure": "#dd6b20"}
    x = np.arange(len(methods))
    width = 0.24
    figure, axis = plt.subplots(figsize=(13, 5.6), constrained_layout=True)
    for index, protocol in enumerate(protocols):
        rates = []
        lower = []
        upper = []
        for method in methods:
            value = report["methods"][method]["protocols"][protocol]["success"]
            rate = float(value["rate"])
            interval = value["wilson_95"]
            rates.append(rate)
            lower.append(rate - float(interval[0]))
            upper.append(float(interval[1]) - rate)
        axis.bar(
            x + (index - 1) * width,
            rates,
            width,
            yerr=np.asarray([lower, upper]),
            capsize=3,
            label=labels[protocol],
            color=colors[protocol],
        )
    axis.set_xticks(
        x,
        [method.replace("gr00t_", "").replace("_", "\n").upper() for method in methods],
    )
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Success rate with Wilson 95% CI")
    axis.set_title("HumanoidArena hidden-final primary aggregate (OpenDoor excluded)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _effects_plot(report: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["primary_comparisons"]
    labels = [row["family"].replace("_contribution", "").replace("_", "\n") for row in rows]
    values = [100.0 * float(row["paired_risk_difference"]) for row in rows]
    colors = ["#2f855a" if value >= 0 else "#c53030" for value in values]
    figure, axis = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    bars = axis.bar(labels, values, color=colors)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Paired success-rate difference (percentage points)")
    axis.set_title("Four predeclared effects; two-sided exact McNemar + Holm")
    axis.grid(axis="y", alpha=0.25)
    for bar, row in zip(bars, rows, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():+.1f} pp\nHolm p={row['holm_adjusted_p']:.3g}",
            ha="center",
            va="bottom" if bar.get_height() >= 0 else "top",
            fontsize=9,
        )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _efficiency_plot(report: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(report["methods"])
    latency = [
        report["methods"][method]["protocols"]["online_failure"]["efficiency"][
            "policy_request_mean_ms"
        ]
        for method in methods
    ]
    wall = [
        report["methods"][method]["protocols"]["online_failure"]["efficiency"][
            "episode_wall_mean_s"
        ]
        for method in methods
    ]
    labels = [method.replace("gr00t_", "").replace("_", "\n").upper() for method in methods]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, values, title, ylabel, color in (
        (axes[0], latency, "Policy request latency", "milliseconds", "#2b6cb0"),
        (axes[1], wall, "Episode wall time", "seconds", "#4a5568"),
    ):
        bars = axis.bar(labels, values, color=color)
        axis.bar_label(bars, fmt="%.1f")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    hardware = report.get("hardware")
    subtitle = "Exact request timers and runner telemetry"
    if hardware:
        subtitle += (
            f" — CPU {hardware['cpu_busy_percent']['mean']:.1f}%, "
            f"GPU {hardware['gpu_utilization_percent']['mean']:.1f}% mean"
        )
    figure.suptitle(subtitle, fontsize=10)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "_artifacts/HumanoidArena/internal-benchmark/plans/hidden_final.plan.json",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=ROOT / "_artifacts/HumanoidArena/internal-benchmark/runs/hidden_final",
    )
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=ROOT / "config/humanoidarena-internal-protocol.lock.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/humanoidarena/internal-hidden-final",
    )
    args = parser.parse_args()
    plan = load_json(args.plan.resolve())
    lock = load_json(args.protocol_lock.resolve())
    runs_root = args.runs_root.resolve()
    records = []
    for method in INTERNAL_METHODS:
        records.extend(load_jsonl(runs_root / method / "records.jsonl"))
    telemetry = _load_jsonl_unvalidated(runs_root / "hardware-telemetry.jsonl")
    report = build_internal_report(plan, records, lock, telemetry)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "report.json", report)
    _success_plot(report, output / "success_and_recovery.png")
    _effects_plot(report, output / "primary_effects.png")
    _efficiency_plot(report, output / "efficiency.png")
    print(
        json.dumps(
            {
                "report_sha256": report["report_sha256"],
                "episode_records": report["episode_records"],
                "claim_status": report["claim_status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
