from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "st_rt_validation_evidence",
        ROOT / "scripts/render_humanoidarena_st_rt_validation.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selection(offset: float) -> dict:
    candidates = {}
    for step in (100, 200, 300):
        candidates[str(step)] = {
            "seeds": {
                str(seed): {"macro_task_mse": 0.3 + seed * 0.01 + offset}
                for seed in (0, 1, 2)
            }
        }
    return {
        "selection_sha256": "a" * 64,
        "selected_step": 300,
        "hidden_test_accessed": False,
        "candidates": candidates,
    }


def test_report_is_paired_and_claim_bounded() -> None:
    report = _module().build_report(
        _selection(0.0), _selection(-0.01), common_manifest="b" * 64, st_manifest="c" * 64
    )
    assert report["hidden_test_accessed"] is False
    assert "open-loop" in report["claim_boundary"]
    assert report["selected_comparison"]["mean_paired_reduction_percent"] > 0
    assert len(report["audit_sha256"]) == 64


def test_report_rejects_hidden_access() -> None:
    common = _selection(0.0)
    common["hidden_test_accessed"] = True
    with pytest.raises(ValueError, match="hidden"):
        _module().build_report(
            common, _selection(-0.01), common_manifest="b" * 64, st_manifest="c" * 64
        )
