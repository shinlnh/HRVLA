from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_humanoidarena_baseline_matrix_fast.py"
SPEC = importlib.util.spec_from_file_location("humanoidarena_matrix_fast", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FAST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FAST)


def test_pending_repeat_ids_reuses_atomic_episode_evidence(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    for repeat_idx, reason in ((0, "success"), (2, "timeout"), (3, "interrupted")):
        (episodes / f"episode-{repeat_idx}.json").write_text(
            json.dumps(
                {
                    "seed": 1,
                    "repeat_idx": repeat_idx,
                    "success": reason == "success",
                    "failure_reason": reason,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    assert FAST.pending_repeat_ids(tmp_path, seed=1, repeats=4) == [1, 3]


def test_episode_seed_matches_locked_upstream_formula() -> None:
    assert FAST._derive_episode_seed("task", 0, 0) == 659473100
    assert FAST._derive_episode_seed("task", 2, 19) == 1207044323


def test_sim_command_preserves_locked_runtime_contract(tmp_path: Path) -> None:
    command = FAST._sim_command(
        task_name="boxing",
        mode="base_test",
        batch_path=tmp_path / "batch.json",
        port=18443,
        record_video_every_n=10,
        step_log_every_n=250,
    )
    assert command[command.index("--device") + 1] == "cuda:0"
    assert command[command.index("--sonic_vla_root_rot6d_layout") + 1] == "row"
    assert command[command.index("--record_video_every_n") + 1] == "10"
    assert "--episode_batch_json" in command


def test_default_fast_matrix_keeps_full_paper_sample_size() -> None:
    assert len(FAST.TASKS) * len(FAST.MODES) * len(FAST.SEEDS) * FAST.REPEATS == 1680
