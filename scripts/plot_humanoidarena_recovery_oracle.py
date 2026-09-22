#!/usr/bin/env python3
"""Render the frozen PI0.5 recovery oracle's complete outcomes, including failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/benchmark/recovery/oracle_pi05_int8_v0_audit.json"
DEFAULT_OUTPUT = ROOT / "results/benchmark/recovery/oracle_pi05_int8_v0_audit.png"


def plot(report: dict, output: Path) -> None:
    if not report.get("run_complete") or report.get("trials") != 180 or report.get("scenarios") != 9:
        raise ValueError("oracle audit is not a complete 180-trial result")
    rows = report["by_scenario"]
    names = [row["scenario_id"].replace("-", " ") for row in rows]
    successes = np.array([row["successes"] for row in rows])
    timeouts = np.array([row["reason_counts"].get("timeout", 0) for row in rows])
    falls = np.array([row["reason_counts"].get("fall", 0) for row in rows])
    other = 20 - successes - timeouts - falls
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.barh(y, successes, color="#25826c", label="Success")
    ax.barh(y, timeouts, left=successes, color="#d1a14d", label="Timeout")
    ax.barh(y, falls, left=successes + timeouts, color="#aa4d5c", label="Fall")
    if np.any(other):
        ax.barh(y, other, left=successes + timeouts + falls, color="#626a76", label="Other behavioral failure")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 21.7)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Independent oracle trials (20 per scenario)")
    ax.axvline(20, color="#283647", linestyle="--", linewidth=1.1, label="Locked 20/20 admission")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    for index, row in enumerate(rows):
        ax.text(20.18, index, f"{row['successes']}/20", va="center", fontsize=9,
                color="#247662" if row["admitted"] else "#8f3d48")
    ax.legend(loc="lower right", ncol=4, bbox_to_anchor=(1, 1.01), fontsize=9)
    fig.suptitle("PI0.5 independent recovery witness · frozen oracle v0", y=0.98, fontsize=14)
    fig.text(
        0.5, 0.035,
        f"Complete evidence: {report['trials']}/180 trials, {report['successes']}/180 successes, "
        f"{report['scenarios_admitted']}/9 scenarios admitted. "
        "Failures are retained; this is not an internal method comparison.",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(left=0.25, right=0.91, top=0.84, bottom=0.13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, metadata={"Software": "HRVLA recovery oracle audit plotter"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(json.loads(args.input.read_text(encoding="utf-8")), args.output)
    print(f"[recovery-oracle-plot] output={args.output}")


if __name__ == "__main__":
    main()
