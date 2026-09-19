from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from hrvla_bench.humanoidarena_bridge import (
    ACTION_DIM,
    ACTION_FIELDS,
    STATE_DIM,
    STATE_FIELDS,
    deterministic_split,
    deterministic_train_validation_hidden_split,
    flatten_action,
    modality_metadata,
    split_action,
    split_state,
    validate_release_info,
)


def _release_info() -> dict:
    return {
        "features": {
            "observation.images.front": {"shape": [480, 640, 3]},
            "observation.state": {"shape": [64]},
            "action": {"shape": [40]},
        },
        "vla_protocol": {
            "schema": "unitree_g1_gmt_refpose_v3_1",
            "version": "3.1",
            "action_dim": 40,
            "action_layout": "root_xy_delta_z_rot6d_joints29_hands2",
            "rotation_6d_layout": "row",
            "action_semantics": "reference_pose_not_robot_current_residual",
            "root_xy_delta_frame": "current_reference_base_frame",
            "root_rotation_frame": "episode_reference_frame",
            "backend_source": "sonic",
        },
    }


def test_bridge_layout_is_contiguous_and_lossless() -> None:
    assert sum(field.width for field in STATE_FIELDS) == STATE_DIM
    assert sum(field.width for field in ACTION_FIELDS) == ACTION_DIM
    state = list(range(STATE_DIM))
    action = list(range(ACTION_DIM))
    assert [value for group in split_state(state).values() for value in group] == state
    assert flatten_action(split_action(action)) == action


def test_bridge_rejects_wrong_width_and_group_schema() -> None:
    with pytest.raises(ValueError, match="expected vector width 64"):
        split_state([0.0] * 63)
    malformed = split_action([0.0] * ACTION_DIM)
    malformed.pop("hand_binary")
    with pytest.raises(ValueError, match="group mismatch"):
        flatten_action(malformed)


def test_release_contract_rejects_semantic_drift() -> None:
    info = _release_info()
    validate_release_info(info)
    changed = copy.deepcopy(info)
    changed["vla_protocol"]["rotation_6d_layout"] = "column"
    with pytest.raises(ValueError, match="incompatible HumanoidArena VLA protocol"):
        validate_release_info(changed)


def test_split_is_per_task_deterministic_disjoint_and_complete() -> None:
    first = deterministic_split(range(100), 0.2, 20260915)
    second = deterministic_split(range(100), 0.2, 20260915)
    assert first == second
    train, heldout = first
    assert len(train) == 80
    assert len(heldout) == 20
    assert not set(train) & set(heldout)
    assert set(train) | set(heldout) == set(range(100))


def test_three_way_split_preserves_hidden_membership_and_freezes_dev() -> None:
    train, validation, hidden = deterministic_train_validation_hidden_split(
        range(100), 0.1, 0.2, 20260915
    )
    old_train, old_hidden = deterministic_split(range(100), 0.2, 20260915)
    assert len(train) == 70
    assert len(validation) == 10
    assert len(hidden) == 20
    assert hidden == old_hidden
    assert set(train) | set(validation) == set(old_train)
    assert not set(train) & set(validation)
    assert not set(train) & set(hidden)
    assert not set(validation) & set(hidden)


def test_modality_metadata_matches_named_bridge_fields() -> None:
    metadata = modality_metadata()
    assert list(metadata["state"]) == [field.name for field in STATE_FIELDS]
    assert list(metadata["action"]) == [field.name for field in ACTION_FIELDS]
    assert metadata["video"]["front"]["original_key"] == "observation.images.front"
    assert metadata["annotation"]["human.task_description"]["original_key"] == "task_index"


def test_materialized_view_keeps_packed_video_as_offset_symlinks(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts/prepare_humanoidarena_gr00t_view.py"
    spec = importlib.util.spec_from_file_location("prepare_ha_view", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dataset = tmp_path / "source/HSI_fixture/sonic_refpose_v3_1"
    (dataset / "meta/episodes/chunk-000").mkdir(parents=True)
    (dataset / "data/chunk-000").mkdir(parents=True)
    video = dataset / "videos/observation.images.front/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"immutable-packed-video-fixture")
    info = _release_info()
    info.update(
        {
            "fps": 50,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": (
                "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            ),
        }
    )
    (dataset / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["Fixture task"], name="task")).to_parquet(
        dataset / "meta/tasks.parquet"
    )

    data_rows = []
    episode_rows = []
    for episode_index in range(5):
        for frame_index in range(2):
            data_rows.append(
                {
                    "observation.state": np.full(64, episode_index, dtype=np.float32),
                    "action": np.full(40, frame_index, dtype=np.float32),
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "index": episode_index * 2 + frame_index,
                    "task_index": 0,
                    "timestamp": frame_index / 50.0,
                }
            )
        start = episode_index * 2 / 50.0
        episode_rows.append(
            {
                "episode_index": episode_index,
                "length": 2,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "videos/observation.images.front/chunk_index": 0,
                "videos/observation.images.front/file_index": 0,
                "videos/observation.images.front/from_timestamp": start,
                "videos/observation.images.front/to_timestamp": start + 2 / 50.0,
            }
        )
    pd.DataFrame(data_rows).to_parquet(dataset / "data/chunk-000/file-000.parquet")
    pd.DataFrame(episode_rows).to_parquet(
        dataset / "meta/episodes/chunk-000/file-000.parquet"
    )

    output = tmp_path / "view"
    manifest = module.materialize_view(
        tmp_path / "source", output, heldout_fraction=0.4, seed=17
    )
    assert manifest["outputs"]["train"]["episodes"] == 3
    assert manifest["outputs"]["heldout"]["episodes"] == 2
    all_rows = []
    for split in ("train", "heldout"):
        rows = [
            json.loads(line)
            for line in (output / split / "meta/episodes.jsonl").read_text().splitlines()
        ]
        all_rows.extend(rows)
        for row in rows:
            link = (
                output
                / split
                / "videos/chunk-000/observation.images.front"
                / f"episode_{row['episode_index']:06d}.mp4"
            )
            assert link.is_symlink()
            assert link.resolve() == video.resolve()
            assert row["video_frame_offset"] == row["source_episode_index"] * 2
    assert {row["source_episode_index"] for row in all_rows} == set(range(5))
