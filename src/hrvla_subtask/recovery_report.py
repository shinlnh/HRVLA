"""Aggregate recovery stress cells into auditable tables and figures."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterable


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cell(path: Path) -> dict[str, Any]:
    summary = _read_json(path / "summary.json")
    manifest = _read_json(path / "manifest.json")
    rate = float(manifest["arguments"]["disturbance_rate"])
    methods = summary.get("methods", {})
    if not methods:
        raise ValueError(f"{path}: summary has no methods")
    return {
        "path": str(path),
        "disturbance_rate": rate,
        "git_revision": manifest["git_revision"],
        "arguments": manifest["arguments"],
        "methods": methods,
    }


def _read_raw_rows(cell: dict[str, Any]) -> Iterable[dict[str, Any]]:
    root = Path(cell["path"])
    for method in cell["methods"]:
        path = root / "raw" / f"{method}.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)


def _task_rates(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[float, str, str], list[int]] = {}
    for cell in cells:
        rate = float(cell["disturbance_rate"])
        for row in _read_raw_rows(cell):
            key = (rate, str(row["method"]), str(row["task_id"]))
            aggregate = counts.setdefault(key, [0, 0, 0, 0])
            aggregate[0] += 1
            aggregate[1] += int(bool(row["success"]))
            aggregate[2] += int(row["failures"])
            aggregate[3] += int(row["recovered_failures"])
    return [
        {
            "disturbance_rate": rate,
            "method": method,
            "task_id": task,
            "episodes": values[0],
            "task_success_rate": values[1] / values[0],
            "failure_recovery_rate": values[3] / values[2] if values[2] else None,
        }
        for (rate, method, task), values in sorted(counts.items())
    ]


def _scalar_rows(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        for method, values in cell["methods"].items():
            row = {
                "disturbance_rate": cell["disturbance_rate"],
                "method": method,
                **{
                    key: value
                    for key, value in values.items()
                    if not isinstance(value, (dict, list))
                },
            }
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    leading = [key for key in ("disturbance_rate", "method", "task_id") if key in keys]
    keys = leading + [key for key in keys if key not in leading]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot(cells: list[dict[str, Any]], task_rows: list[dict[str, Any]], output: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required to render aggregate charts") from exc

    methods = sorted(cells[0]["methods"])
    rates = [float(cell["disturbance_rate"]) for cell in cells]
    labels = {
        "agentchord": "AgentChord proxy",
        "baton_str": "BATON-STR",
        "doremi": "DoReMi proxy",
        "inner_monologue": "Inner Monologue proxy",
        "no_recovery": "No recovery",
        "rekep": "ReKep proxy",
    }
    palette = {
        "agentchord": "#319795",
        "baton_str": "#2b6cb0",
        "doremi": "#d69e2e",
        "inner_monologue": "#805ad5",
        "no_recovery": "#718096",
        "rekep": "#c53030",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for axis, (metric, title) in zip(
        axes,
        (
            ("task_success_rate", "Task success under disturbance"),
            ("failure_recovery_rate", "Failure recovery under disturbance"),
        ),
    ):
        for method in methods:
            values = [cell["methods"][method][metric] or 0.0 for cell in cells]
            axis.plot(
                rates,
                values,
                marker="o",
                linewidth=2.2 if method == "baton_str" else 1.4,
                color=palette.get(method),
                label=labels.get(method, method),
            )
        axis.set_xlabel("Injected disturbance probability")
        axis.set_ylabel("Rate")
        axis.set_ylim(0.0, 1.03)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="lower left")
    robustness = output / "recovery_robustness.png"
    fig.savefig(robustness, dpi=180)
    plt.close(fig)

    standard = min(cells, key=lambda item: abs(float(item["disturbance_rate"]) - 0.15))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for method in methods:
        values = standard["methods"][method]
        style = {
            "color": palette.get(method),
            "s": 110 if method == "baton_str" else 70,
            "label": labels.get(method, method),
        }
        axes[0].scatter(values["mean_actions"], values["task_success_rate"], **style)
        axes[1].scatter(
            values["monitor_queries_per_failure"] or 0.0,
            values["failure_recovery_rate"] or 0.0,
            **style,
        )
    axes[0].set(
        xlabel="Mean actions (lower is better)",
        ylabel="Task success",
        title="Success/action Pareto view",
    )
    axes[1].set(
        xlabel="Monitor queries per failure",
        ylabel="Failure recovery",
        title="Recovery/monitor trade-off",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="lower center", ncol=2)
    pareto = output / "recovery_pareto.png"
    fig.savefig(pareto, dpi=180)
    plt.close(fig)

    standard_rate = float(standard["disturbance_rate"])
    selected = [row for row in task_rows if row["disturbance_rate"] == standard_rate]
    tasks = sorted({str(row["task_id"]) for row in selected})
    lookup = {
        (str(row["method"]), str(row["task_id"])): float(row["task_success_rate"])
        for row in selected
    }
    matrix = [[lookup[(method, task)] for task in tasks] for method in methods]
    fig, axis = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    axis.set_xticks(range(len(tasks)), [task.replace("_", "\n") for task in tasks], fontsize=8)
    axis.set_yticks(range(len(methods)), [labels.get(method, method) for method in methods])
    axis.set_title(f"Per-task success at disturbance={standard_rate:.2f}")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, f"{value:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axis, label="Task success rate")
    heatmap = output / "recovery_task_heatmap.png"
    fig.savefig(heatmap, dpi=180)
    plt.close(fig)
    return [robustness, pareto, heatmap]


def aggregate_recovery_cells(
    cell_paths: Iterable[str | Path], output_dir: str | Path
) -> dict[str, Any]:
    cells = sorted(
        (_load_cell(Path(path).resolve()) for path in cell_paths),
        key=lambda item: item["disturbance_rate"],
    )
    if len(cells) < 2:
        raise ValueError("at least two recovery cells are required")
    method_sets = {tuple(sorted(cell["methods"])) for cell in cells}
    if len(method_sets) != 1:
        raise ValueError("all recovery cells must contain the same methods")
    revisions = sorted({str(cell["git_revision"]) for cell in cells})
    task_rows = _task_rates(cells)
    scalar_rows = _scalar_rows(cells)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "recovery_robustness.csv", scalar_rows)
    _write_csv(output / "recovery_per_task.csv", task_rows)
    charts = _plot(cells, task_rows, output)
    result = {
        "schema_version": 1,
        "git_revisions": revisions,
        "cells": cells,
        "artifacts": [path.name for path in charts],
    }
    (output / "recovery_comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
