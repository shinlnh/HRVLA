from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "run_humanoidarena_rt_validation_test",
        ROOT / "scripts/run_humanoidarena_rt_validation.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_write_is_idempotent_but_rejects_drift(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "selection.json"
    value = {"selected_step": 300, "selection_sha256": "a" * 64}
    module._write_once(path, value)
    module._write_once(path, value)
    assert json.loads(path.read_text()) == value
    with pytest.raises(FileExistsError, match="different"):
        module._write_once(path, {**value, "selected_step": 200})
