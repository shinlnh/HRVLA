from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_benchmark_evidence.py"
SPEC = importlib.util.spec_from_file_location("benchmark_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EVIDENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVIDENCE)


def test_readiness_keeps_incomparable_workstreams_separate() -> None:
    rows = EVIDENCE.readiness_rows(
        {"matrix": {"cells_complete": 2, "cells_expected": 84}}
    )
    by_name = {row["workstream"]: row for row in rows}
    assert by_name["Planner component"]["completed"] == 5
    assert by_name["PI0.5 + SONIC"]["completed"] == 2
    assert by_name["Scenario admission"]["completed"] == 0
    assert "overall" not in by_name


def test_storage_cleanup_evidence_balances_reclaimed_space() -> None:
    import json

    summary = json.loads(
        (ROOT / "results/benchmark/storage/cleanup.json").read_text(encoding="utf-8")
    )
    assert summary["free_space"]["after"] - summary["free_space"]["before"] == summary["reclaimed_total"]
    assert "not benchmark metrics" in summary["claim_boundary"]


def test_readiness_uses_audited_dynamic_completion_inputs() -> None:
    checkpoints = {
        "status": "ready_for_frozen_execution",
        "checkpoints": {
            method: {
                str(seed): {
                    "published_revision": "a" * 40,
                    "manifest_sha256": "b" * 64,
                }
                for seed in range(3)
            }
            for method in ("gr00t_st_rt", "gr00t_str_rt")
        },
    }
    suite = {
        "tasks": [
            {"scenarios": [{"admission": {"status": "admitted"}} for _ in range(9)]}
        ]
    }
    progress = {
        "records_expected": 50,
        "methods": {
            method: 10
            for method in (
                "gr00t_sonic",
                "gr00t_st",
                "gr00t_st_rt",
                "gr00t_str",
                "gr00t_str_rt",
            )
        },
    }
    rows = EVIDENCE.readiness_rows(None, checkpoints, suite, progress)
    by_name = {row["workstream"]: row for row in rows}
    assert by_name["HA 40-D RT training"]["completed"] == 6
    assert by_name["Scenario admission"]["completed"] == 9
    assert by_name["Internal closed loop"]["completed"] == 5
