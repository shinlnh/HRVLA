from __future__ import annotations

import json
from pathlib import Path

import pytest

from hrvla_bench.paper_results import CONDITIONS, load_metrics, summarize


def _write_metrics(path: Path, mse: float) -> None:
    rows = [
        {"trajectory_id": trajectory, "condition": condition, "mse": mse}
        for condition in CONDITIONS
        for trajectory in range(42)
    ]
    payload = {
        "conditions": list(CONDITIONS),
        "trajectory_ids": list(range(42)),
        "aggregates": {condition: {"mse": mse} for condition in CONDITIONS},
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summary_preserves_seed_level_replication(tmp_path: Path) -> None:
    for seed in range(3):
        _write_metrics(tmp_path / f"st-seed-{seed}/metrics.json", 0.10 + seed * 0.01)
        _write_metrics(tmp_path / f"str-seed-{seed}/metrics.json", 0.09 + seed * 0.01)
    summary = summarize(tmp_path, [0, 1, 2])
    clean = summary["paired_str_rt_minus_st_rt"]["clean"]
    assert clean["seed_count"] == 3
    assert clean["mean_mse_reduction"] == pytest.approx(0.01)
    assert clean["improved_seed_count"] == 3


def test_metrics_require_full_paired_protocol(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    _write_metrics(path, 0.1)
    payload = json.loads(path.read_text())
    payload["rows"].pop()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="210"):
        load_metrics(path)
