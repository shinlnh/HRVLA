from __future__ import annotations

import numpy as np
import pytest

from hrvla_bench.recovery_demonstration import CompactRecoveryDemonstration


def _recorder() -> CompactRecoveryDemonstration:
    return CompactRecoveryDemonstration(
        scenario_id="drop",
        task_id="pick",
        episode_seed=17,
        instruction="Reacquire and place the object.",
        method_program_sha256="a" * 64,
        control_dt_s=0.02,
    )


def test_compact_recovery_demonstration_is_dimension_checked_and_hashed(tmp_path) -> None:
    video = tmp_path / "evidence.mp4"
    video.write_bytes(b"video")
    recorder = _recorder()
    for index in range(40):
        recorder.append(
            np.full(64, index, dtype=np.float32),
            np.full(40, index, dtype=np.float32),
            video_frame_index=index + 9,
            control_step=index + 8,
        )
    manifest = recorder.finalize(
        tmp_path, success=True, failure_reason="success", video_path=str(video)
    )
    assert manifest["eligible_for_recovery_training"] is True
    assert manifest["frames"] == 40
    assert len(manifest["arrays_sha256"]) == len(manifest["manifest_sha256"]) == 64
    arrays = np.load(tmp_path / "recovery-demonstration.npz")
    assert arrays["observation_state"].shape == (40, 64)
    assert arrays["action"].shape == (40, 40)


def test_compact_recovery_demonstration_rejects_wrong_action_contract() -> None:
    with pytest.raises(ValueError, match="action shape"):
        _recorder().append(np.zeros(64), np.zeros(43), video_frame_index=0, control_step=0)


def test_failed_demonstration_is_preserved_but_not_training_eligible(tmp_path) -> None:
    recorder = _recorder()
    recorder.append(np.zeros(64), np.zeros(40), video_frame_index=0, control_step=0)
    manifest = recorder.finalize(
        tmp_path, success=False, failure_reason="timeout", video_path=""
    )
    assert manifest["eligible_for_recovery_training"] is False
