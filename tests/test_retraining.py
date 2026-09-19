from __future__ import annotations

import json
from pathlib import Path

import pytest

from hrvla_bench.retraining import build_command, is_complete, load_lock, run_directory


ROOT = Path(__file__).resolve().parents[1]


def test_locked_matrix_builds_distinct_seed_outputs() -> None:
    lock = load_lock(ROOT / "config/paper-retraining.lock.json")
    command = build_command(ROOT, Path("python"), lock, "STR-RT", 2)
    assert "--global-batch-size" in command
    assert command[command.index("--global-batch-size") + 1] == "16"
    assert command[command.index("--seed") + 1] == "2"
    assert run_directory(ROOT, lock, "STR-RT", 2).name == "str-seed-2"


def test_completion_requires_exact_final_step(tmp_path: Path) -> None:
    state = tmp_path / "checkpoints/checkpoint-300/trainer_state.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"global_step": 299}), encoding="utf-8")
    assert is_complete(tmp_path, 300) is False
    state.write_text(json.dumps({"global_step": 300}), encoding="utf-8")
    assert is_complete(tmp_path, 300) is True


def test_lock_rejects_changed_seed_contract(tmp_path: Path) -> None:
    lock = json.loads((ROOT / "config/paper-retraining.lock.json").read_text())
    lock["training_seeds"] = [42]
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="seeds"):
        load_lock(path)
