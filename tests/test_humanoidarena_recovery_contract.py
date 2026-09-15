from __future__ import annotations

from pathlib import Path

import pytest

from hrvla_bench.humanoidarena_recovery_contract import resolve_python_symbol


def test_predicate_source_resolver_hashes_definition_without_importing(tmp_path: Path) -> None:
    source = tmp_path / "tasks/task/mdp/rewards.py"
    source.parent.mkdir(parents=True)
    source.write_text("import unavailable_isaac_sim\n\ndef compute_success_mask(env):\n    return True\n")
    result = resolve_python_symbol(tmp_path, "tasks/task/mdp/rewards.py:compute_success_mask")
    assert result["symbol"] == "compute_success_mask"
    assert len(result["source_sha256"]) == 64


def test_predicate_source_resolver_rejects_missing_or_escaping_source(tmp_path: Path) -> None:
    source = tmp_path / "rewards.py"
    source.write_text("def another_function():\n    pass\n")
    with pytest.raises(ValueError, match="missing"):
        resolve_python_symbol(tmp_path, "rewards.py:compute_success_mask")
    with pytest.raises(ValueError, match="inside"):
        resolve_python_symbol(tmp_path, "../rewards.py:another_function")
