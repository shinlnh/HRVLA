from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hrvla_bench.humanoidarena_recovery_contract import resolve_python_symbol
from hrvla_bench.recovery_injector_contract import validate_recovery_injector_contract


ROOT = Path(__file__).resolve().parents[1]
SUITE = json.loads(
    (ROOT / "benchmark/suites/hrvla_recovery_v0.json").read_text(encoding="utf-8")
)


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


def test_all_nine_recovery_injector_contracts_are_fully_typed() -> None:
    rows = validate_recovery_injector_contract(SUITE)
    assert len(rows) == 9
    assert {row["interface_seam"] for row in rows} == {
        "scene",
        "scene+semantic-action40",
        "semantic-action40",
        "post-encoder-latent64",
    }
    assert all(len(row["injector_parameters_sha256"]) == 64 for row in rows)
    assert all(len(row["detector_parameters_sha256"]) == 64 for row in rows)


def test_injector_contract_rejects_underspecified_direction() -> None:
    suite = copy.deepcopy(SUITE)
    football = next(task for task in suite["tasks"] if task["id"] == "football")
    del football["scenarios"][0]["injector"]["parameters"]["lateral_direction_robot"]
    with pytest.raises(ValueError, match="lateral_direction_robot"):
        validate_recovery_injector_contract(suite)


def test_open_door_obstruction_requires_geometry_success() -> None:
    suite = copy.deepcopy(SUITE)
    door = next(task for task in suite["tasks"] if task["id"] == "open_door")
    del door["runtime_environment"]
    with pytest.raises(ValueError, match="require geometry"):
        validate_recovery_injector_contract(suite)
