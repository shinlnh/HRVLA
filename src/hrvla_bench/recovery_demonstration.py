"""Compact, auditable recovery demonstrations captured during oracle rollout."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .humanoidarena_bridge import ACTION_DIM, STATE_DIM
from .plan import canonical_sha256


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CompactRecoveryDemonstration:
    """Buffer state64/action40 pairs only after the locked failure is active."""

    def __init__(
        self,
        *,
        scenario_id: str,
        task_id: str,
        episode_seed: int,
        instruction: str,
        method_program_sha256: str,
        control_dt_s: float,
    ) -> None:
        if not instruction.strip():
            raise ValueError("recovery demonstration instruction is empty")
        self.scenario_id = scenario_id
        self.task_id = task_id
        self.episode_seed = int(episode_seed)
        self.instruction = instruction
        self.method_program_sha256 = method_program_sha256
        self.control_dt_s = float(control_dt_s)
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.video_frame_indices: list[int] = []
        self.control_steps: list[int] = []

    def append(
        self,
        state: Any,
        action: Any,
        *,
        video_frame_index: int,
        control_step: int,
    ) -> None:
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if state_array.shape != (STATE_DIM,):
            raise ValueError(f"recovery demonstration state shape is {state_array.shape}")
        if action_array.shape != (ACTION_DIM,):
            raise ValueError(f"recovery demonstration action shape is {action_array.shape}")
        if not np.isfinite(state_array).all() or not np.isfinite(action_array).all():
            raise ValueError("recovery demonstration contains a non-finite value")
        if self.control_steps and int(control_step) <= self.control_steps[-1]:
            raise ValueError("recovery demonstration control steps are not increasing")
        if self.video_frame_indices and int(video_frame_index) <= self.video_frame_indices[-1]:
            raise ValueError("recovery demonstration video frames are not increasing")
        self.states.append(state_array.copy())
        self.actions.append(action_array.copy())
        self.video_frame_indices.append(int(video_frame_index))
        self.control_steps.append(int(control_step))

    def finalize(
        self,
        output_dir: Path,
        *,
        success: bool,
        failure_reason: str,
        video_path: str,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays_path = output_dir / "recovery-demonstration.npz"
        temporary = arrays_path.with_suffix(".npz.tmp")
        states = (
            np.stack(self.states).astype(np.float32, copy=False)
            if self.states
            else np.empty((0, STATE_DIM), dtype=np.float32)
        )
        actions = (
            np.stack(self.actions).astype(np.float32, copy=False)
            if self.actions
            else np.empty((0, ACTION_DIM), dtype=np.float32)
        )
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                observation_state=states,
                action=actions,
                video_frame_index=np.asarray(self.video_frame_indices, dtype=np.int64),
                control_step=np.asarray(self.control_steps, dtype=np.int64),
            )
        os.replace(temporary, arrays_path)
        resolved_video = Path(video_path).resolve() if video_path else None
        video_exists = bool(resolved_video and resolved_video.is_file())
        eligible = bool(success and len(states) >= 40 and video_exists)
        core = {
            "schema_version": 1,
            "claim_boundary": (
                "PI0.5 behavior relabelled with the frozen recovery instruction; "
                "training-only and never an internal evaluation result"
            ),
            "task_id": self.task_id,
            "scenario_id": self.scenario_id,
            "episode_seed": self.episode_seed,
            "instruction": self.instruction,
            "method_program_sha256": self.method_program_sha256,
            "control_dt_s": self.control_dt_s,
            "frames": len(states),
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "failure_active_at_first_frame": True,
            "success": bool(success),
            "failure_reason": str(failure_reason),
            "eligible_for_recovery_training": eligible,
            "arrays_path": arrays_path.name,
            "arrays_sha256": _file_sha256(arrays_path),
            "video_path": "" if resolved_video is None else str(resolved_video),
            "video_sha256": _file_sha256(resolved_video) if video_exists else None,
            "video_frame_indices": {
                "first": self.video_frame_indices[0] if self.video_frame_indices else None,
                "last": self.video_frame_indices[-1] if self.video_frame_indices else None,
                "count": len(self.video_frame_indices),
            },
        }
        manifest = {**core, "manifest_sha256": canonical_sha256(core)}
        manifest_path = output_dir / "recovery-demonstration.json"
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_manifest, manifest_path)
        return manifest


__all__ = ["CompactRecoveryDemonstration"]
