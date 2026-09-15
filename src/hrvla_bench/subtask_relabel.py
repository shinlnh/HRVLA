"""Deterministic weak temporal labels for HumanoidArena subtask post-training.

These labels are training supervision only.  Closed-loop evaluation continues
to use the source-backed simulator predicates in ``humanoidarena_method_runtime``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .humanoidarena_bridge import ACTION_DIM, STATE_DIM


@dataclass(frozen=True)
class Segmentation:
    labels: np.ndarray
    audit: dict[str, Any]


def _first_sustained(values: np.ndarray, start: int, stop: int, width: int = 5) -> int | None:
    run = 0
    for index in range(max(0, start), min(len(values), stop)):
        run = run + 1 if bool(values[index]) else 0
        if run >= width:
            return index - width + 1
    return None


def _motion_onset(energy: np.ndarray, start: int, stop: int) -> int | None:
    segment = np.asarray(energy[start:stop], dtype=np.float64)
    if len(segment) < 10 or not np.isfinite(segment).all():
        return None
    width = min(15, max(3, len(segment) // 50))
    smoothed = np.convolve(segment, np.ones(width) / width, mode="same")
    peak = int(np.argmax(smoothed))
    median = float(np.median(smoothed))
    mad = float(np.median(np.abs(smoothed - median)))
    threshold = median + max(1e-6, 2.0 * mad)
    if float(smoothed.max()) < threshold:
        return None
    above = smoothed >= threshold
    onset = peak
    while onset > 0 and above[onset - 1]:
        onset -= 1
    return start + onset


def _valid_boundary(value: int | None, low: int, high: int) -> bool:
    return value is not None and low <= value <= high


def _fallback_boundaries(length: int, skills: int) -> list[int]:
    return [round(length * index / skills) for index in range(1, skills)]


def segment_subtasks(
    task_key: str,
    states: Any,
    actions: Any,
    *,
    skill_count: int,
    minimum_phase_frames: int = 40,
) -> Segmentation:
    """Return monotonic per-frame skill labels and explicit proxy diagnostics."""

    state = np.asarray(states, dtype=np.float32)
    action = np.asarray(actions, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != STATE_DIM:
        raise ValueError(f"expected state matrix [N, {STATE_DIM}]")
    if action.ndim != 2 or action.shape != (len(state), ACTION_DIM):
        raise ValueError(f"expected action matrix [N, {ACTION_DIM}]")
    if skill_count not in {2, 3}:
        raise ValueError("HumanoidArena programs currently require two or three skills")
    if len(state) < skill_count * minimum_phase_frames:
        raise ValueError("episode is too short to supervise every subtask")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("episode contains non-finite state/action values")

    length = len(state)
    low = minimum_phase_frames
    high = length - minimum_phase_frames
    hands_closed = np.max(action[:, 38:40], axis=1) >= 0.5
    first_close = _first_sustained(hands_closed, low, high)
    first_release = None
    if first_close is not None:
        first_release = _first_sustained(
            ~hands_closed,
            first_close + minimum_phase_frames,
            high,
        )
    lower_body_energy = np.linalg.norm(state[:, 35:49], axis=1)
    upper_body_energy = np.linalg.norm(state[:, 49:64], axis=1)
    proxy = ""
    candidates: list[int | None]

    if task_key in {"doubledesk", "pp_box", "open_door"}:
        proxy = "hand-binary-close-release"
        candidates = [first_close]
        if skill_count == 3:
            candidates.append(first_release)
    elif task_key == "vision_navi":
        proxy = "integrated-root-reference-displacement"
        distance = np.cumsum(np.linalg.norm(action[:, 0:2], axis=1))
        reached = np.flatnonzero(distance >= 1.0)
        candidates = [int(reached[0]) if len(reached) else None]
    elif task_key == "sit_sofa":
        proxy = "root-height-descent-onset"
        baseline = float(np.median(action[: max(low, length // 10), 2]))
        deviations = np.abs(action[: max(low, length // 10), 2] - baseline)
        threshold = baseline - max(0.03, 3.0 * float(np.median(deviations)))
        candidates = [_first_sustained(action[:, 2] <= threshold, low, high)]
    elif task_key == "football":
        proxy = "lower-body-motion-burst-onset"
        candidates = [_motion_onset(lower_body_energy, low, high)]
    elif task_key == "boxing":
        proxy = "upper-body-motion-burst-onset"
        candidates = [_motion_onset(upper_body_energy, low, high)]
    else:
        raise ValueError(f"no frozen subtask training proxy for task key {task_key!r}")

    fallback_reasons = []
    boundaries: list[int] = []
    previous = 0
    expected_fallback = _fallback_boundaries(length, skill_count)
    for index, candidate in enumerate(candidates):
        local_low = max(low, previous + minimum_phase_frames)
        remaining = len(candidates) - index
        local_high = min(high, length - remaining * minimum_phase_frames)
        if _valid_boundary(candidate, local_low, local_high):
            boundary = int(candidate)
        else:
            boundary = max(local_low, min(local_high, expected_fallback[index]))
            fallback_reasons.append(
                {
                    "boundary_index": index,
                    "candidate": candidate,
                    "allowed": [local_low, local_high],
                    "fallback": boundary,
                }
            )
        boundaries.append(boundary)
        previous = boundary
    if len(boundaries) != skill_count - 1 or boundaries != sorted(set(boundaries)):
        raise AssertionError("subtask boundaries are not strictly monotonic")

    labels = np.searchsorted(np.asarray(boundaries), np.arange(length), side="right")
    counts = np.bincount(labels, minlength=skill_count)
    if np.any(counts < minimum_phase_frames):
        raise AssertionError("validated segmentation produced an undersized phase")
    audit = {
        "schema_version": 1,
        "task_key": task_key,
        "proxy": proxy,
        "claim_boundary": (
            "weak training labels from robot state/action only; not a simulator-event "
            "metric and not evidence that a semantic transition succeeded"
        ),
        "frames": length,
        "skill_count": skill_count,
        "minimum_phase_frames": minimum_phase_frames,
        "boundaries": boundaries,
        "raw_candidates": candidates,
        "phase_frame_counts": counts.tolist(),
        "fallback_used": bool(fallback_reasons),
        "fallback_reasons": fallback_reasons,
    }
    return Segmentation(labels=labels.astype(np.int64), audit=audit)


__all__ = ["Segmentation", "segment_subtasks"]
