from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_humanoidarena_recovery_data.py"
SPEC = importlib.util.spec_from_file_location("recovery_data", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATA)


def test_recovery_split_is_deterministic_disjoint_and_fourteen_three_three() -> None:
    first = DATA.recovery_split("drop", list(range(20)))
    second = DATA.recovery_split("drop", list(reversed(range(20))))
    assert first == second
    assert [len(first[name]) for name in ("train", "validation", "heldout")] == [14, 3, 3]
    assert not (set(first["train"]) & set(first["validation"]))
    assert set().union(*map(set, first.values())) == set(range(20))


def test_recovery_split_rejects_posthoc_trial_removal() -> None:
    with pytest.raises(ValueError, match="0..19"):
        DATA.recovery_split("drop", list(range(19)))
