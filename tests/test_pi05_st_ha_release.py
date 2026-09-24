"""Regression gates for the completed PI0.5-ST HumanoidArena branch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import pytest

from scripts.audit_pi05_st_ha_release import audit, require_hash
from scripts.benchmark_v2_preflight import inspect_contract


ROOT = Path(__file__).resolve().parents[1]


def test_complete_contract_passes_fail_closed_preflight() -> None:
    contract = json.loads((ROOT / "benchmark/v2_contract.json").read_text())
    assert contract["status"] == "complete"
    assert inspect_contract(
        contract,
        ROOT,
        branch="feat(PI05-ST-benchmark-HA)/evaluate",
        architecture_head=contract["architecture_revision"],
        architecture_in_history=True,
        suite_head=contract["suite_revision"],
    ) == []


def test_checked_in_release_passes() -> None:
    assert audit()["checked_in_videos"] == 7


def test_locked_hash_rejects_changed_bytes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "evidence.json"
        path.write_text("original", encoding="utf-8")
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        require_hash(path, expected)
        path.write_text("changed", encoding="utf-8")
        with pytest.raises(ValueError, match="hash-mismatched"):
            require_hash(path, expected)
