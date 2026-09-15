from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_humanoidarena_recovery_episode.py"
SPEC = importlib.util.spec_from_file_location("recovery_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_single_episode_contract_uses_explicit_episode_seed() -> None:
    task, seed = RUNNER._single_episode_contract(
        ["--task", "task-id", "--seed", "3", "--episode_seed", "17", "--headless"]
    )
    assert task == "task-id"
    assert seed == 17


def test_single_episode_contract_rejects_a_multi_episode_batch(tmp_path) -> None:
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps({"episodes": [{"episode_seed": 1}, {"episode_seed": 2}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one"):
        RUNNER._single_episode_contract(
            ["--task", "task-id", "--seed", "3", "--episode_batch_json", str(batch)]
        )


def test_per_episode_output_contract_accepts_a_unique_batch(tmp_path) -> None:
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            {
                "episodes": [
                    {"episode_index": 0, "episode_seed": 11},
                    {"episode_index": 1, "episode_seed": 22},
                ]
            }
        ),
        encoding="utf-8",
    )
    task, seeds = RUNNER._episode_contract(
        ["--task", "task-id", "--seed", "3", "--episode_batch_json", str(batch)],
        allow_batch=True,
    )
    assert task == "task-id"
    assert seeds == [11, 22]


def test_encoder_proxy_perturbs_only_the_first_encoder_output() -> None:
    class Encoder:
        def run(self, *args, **kwargs):
            return [np.ones((1, 64), dtype=np.float32), np.array([5.0])]

    class Runtime:
        def transform_vla_action(self, env, action, *, task_success):
            assert env == "env"
            assert task_success is False
            return action * 0.55

    state = {"runtime": Runtime(), "env": "env", "task_success": False}
    outputs = RUNNER._EncoderProxy(Encoder(), state).run(None)
    np.testing.assert_allclose(outputs[0], 0.55)
    np.testing.assert_array_equal(outputs[1], [5.0])
