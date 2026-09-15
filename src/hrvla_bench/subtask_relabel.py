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


def _segment_cost(prefix: np.ndarray, prefix_sq: np.ndarray, start: int, stop: int) -> float:
    count = stop - start
    total = prefix[stop] - prefix[start]
    total_sq = prefix_sq[stop] - prefix_sq[start]
    return float(np.sum(total_sq - total * total / count))


def _ordered_change_points(
    state: np.ndarray,
    action: np.ndarray,
    *,
    segments: int,
    minimum_phase_frames: int,
) -> tuple[list[int] | None, dict[str, Any]]:
    """Exact-order dynamic programming over compact robot-motion features."""

    action_delta = np.vstack([np.zeros((1, ACTION_DIM), dtype=np.float32), np.diff(action, axis=0)])
    features = np.column_stack(
        [
            np.linalg.norm(action[:, 0:2], axis=1),
            action[:, 2],
            np.linalg.norm(state[:, 35:49], axis=1),
            np.linalg.norm(state[:, 49:64], axis=1),
            np.linalg.norm(action_delta[:, 9:20], axis=1),
            np.linalg.norm(action_delta[:, 20:38], axis=1),
        ]
    ).astype(np.float64)
    smooth_width = min(15, max(3, len(features) // 100))
    kernel = np.ones(smooth_width, dtype=np.float64) / smooth_width
    features = np.column_stack(
        [np.convolve(features[:, column], kernel, mode="same") for column in range(features.shape[1])]
    )
    median = np.median(features, axis=0)
    scale = np.quantile(features, 0.75, axis=0) - np.quantile(features, 0.25, axis=0)
    informative = scale > 1e-6
    if not informative.any():
        return None, {"reason": "no_informative_motion_feature"}
    features = (features[:, informative] - median[informative]) / scale[informative]
    prefix = np.vstack([np.zeros((1, features.shape[1])), np.cumsum(features, axis=0)])
    prefix_sq = np.vstack([np.zeros((1, features.shape[1])), np.cumsum(features**2, axis=0)])
    length = len(features)
    stride = max(1, length // 240)
    candidates = list(range(minimum_phase_frames, length - minimum_phase_frames + 1, stride))
    for value in _fallback_boundaries(length, segments):
        if minimum_phase_frames <= value <= length - minimum_phase_frames:
            candidates.append(value)
    candidates = sorted(set(candidates))
    previous: dict[int, tuple[float, list[int]]] = {
        boundary: (_segment_cost(prefix, prefix_sq, 0, boundary), [boundary])
        for boundary in candidates
    }
    for segment_index in range(2, segments):
        current: dict[int, tuple[float, list[int]]] = {}
        for boundary in candidates:
            choices = [
                (
                    cost + _segment_cost(prefix, prefix_sq, prior, boundary),
                    points + [boundary],
                )
                for prior, (cost, points) in previous.items()
                if boundary - prior >= minimum_phase_frames
            ]
            if choices:
                current[boundary] = min(choices, key=lambda row: (row[0], row[1]))
        previous = current
    choices = [
        (cost + _segment_cost(prefix, prefix_sq, boundary, length), points)
        for boundary, (cost, points) in previous.items()
        if length - boundary >= minimum_phase_frames
    ]
    if not choices:
        return None, {"reason": "no_feasible_change_point_path"}
    best_cost, boundaries = min(choices, key=lambda row: (row[0], row[1]))
    total_cost = _segment_cost(prefix, prefix_sq, 0, length)
    explained = max(0.0, 1.0 - best_cost / max(total_cost, 1e-12))
    diagnostics = {
        "feature_names": [
            name
            for name, keep in zip(
                (
                    "root_xy_reference_motion",
                    "root_height_reference",
                    "lower_body_velocity",
                    "upper_body_velocity",
                    "lower_action_change",
                    "upper_action_change",
                ),
                informative,
            )
            if keep
        ],
        "candidate_stride": stride,
        "within_segment_sse": best_cost,
        "total_sse": total_cost,
        "explained_motion_variance": explained,
    }
    if explained < 0.15:
        return None, {**diagnostics, "reason": "change_points_explain_too_little_variance"}
    return boundaries, diagnostics


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

    proxy_details: dict[str, Any] = {}
    if task_key == "doubledesk":
        proxy = "hand-binary-close-release"
        candidates = [first_close]
    elif task_key in {"pp_box", "open_door"}:
        if first_close is not None and first_release is not None:
            proxy = "hand-binary-close-release"
            candidates = [first_close, first_release]
        else:
            proxy = "ordered-multivariate-change-point"
            change_points, proxy_details = _ordered_change_points(
                state,
                action,
                segments=skill_count,
                minimum_phase_frames=minimum_phase_frames,
            )
            candidates = [None] * (skill_count - 1) if change_points is None else change_points
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
        "proxy_details": proxy_details,
        "phase_frame_counts": counts.tolist(),
        "fallback_used": bool(fallback_reasons),
        "fallback_reasons": fallback_reasons,
    }
    return Segmentation(labels=labels.astype(np.int64), audit=audit)


__all__ = ["Segmentation", "segment_subtasks"]
