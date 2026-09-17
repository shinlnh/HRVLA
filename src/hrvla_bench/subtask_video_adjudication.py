"""Auditable temporal consensus for video-derived subtask labels.

The VLM is only allowed to replace an explicit state/action fallback.  This
module deliberately contains no model runtime so that parsing, consensus, and
dataset admission remain testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class TemporalProposal:
    boundaries: tuple[int, ...]
    confidences: tuple[float, ...]
    evidence: tuple[str, ...]
    raw_text: str


@dataclass(frozen=True)
class TemporalBoundaryProposal:
    frame_index: int
    sample_position: int
    confidence: float
    evidence: str
    raw_text: str


def parse_temporal_boundary_proposal(
    text: str,
    *,
    episode_frames: int,
    sample_indices: Iterable[int],
    allowed_sample_positions: Iterable[int] | None = None,
) -> TemporalBoundaryProposal:
    """Parse one independently grounded transition from a strict JSON object."""

    objects = list(_json_objects(text))
    if not objects:
        raise ValueError("temporal VLM output contains no JSON object")
    value = objects[-1]
    position = value.get("boundary_sample_position")
    confidence = value.get("boundary_confidence")
    evidence = value.get("visible_evidence")
    if isinstance(position, bool) or not isinstance(position, int):
        raise ValueError("boundary_sample_position must be one JSON integer")
    samples = tuple(int(item) for item in sample_indices)
    if position < 0 or position >= len(samples):
        raise ValueError("temporal VLM sample position lies outside the contact sheet")
    if allowed_sample_positions is not None:
        allowed = {int(item) for item in allowed_sample_positions}
        if position not in allowed:
            raise ValueError("temporal VLM sample position violates the frozen phase constraints")
    frame_index = samples[position]
    if frame_index <= 0 or frame_index >= episode_frames:
        raise ValueError("temporal VLM boundary lies outside the episode")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("boundary_confidence must be one JSON number")
    numeric_confidence = float(confidence)
    if not 0.0 <= numeric_confidence <= 1.0:
        raise ValueError("boundary confidence lies outside [0, 1]")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("visible_evidence must be one non-empty JSON string")
    return TemporalBoundaryProposal(
        frame_index=frame_index,
        sample_position=position,
        confidence=numeric_confidence,
        evidence=evidence.strip(),
        raw_text=text,
    )


def temporal_boundary_format_repair_prompt(
    original_prompt: str,
    error: str,
    *,
    repair_number: int,
) -> str:
    """Request structural repair for one boundary without supplying a replacement."""

    emphasis = (
        "Do not return lists, markdown, or explanations."
        if repair_number == 1
        else "Use compact single-line JSON with exactly three scalar values."
    )
    return (
        f"{original_prompt}\n\n"
        "FORMAT_CORRECTION: Your previous answer failed strict validation. "
        f"Validation error: {error}. {emphasis} boundary_sample_position must be one "
        "allowed integer, boundary_confidence must be one number from 0 to 1, and "
        "visible_evidence must be one brief non-empty string. This retry corrects "
        "structure only; preserve your visual judgment.\nFINAL_JSON_ONLY:"
    )


def infer_temporal_boundary_with_repair(
    infer: Callable[[str], str],
    prompt: str,
    *,
    episode_frames: int,
    sample_indices: Iterable[int],
    allowed_sample_positions: Iterable[int] | None = None,
    maximum_format_repairs: int,
    record_attempt: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[TemporalBoundaryProposal | None, list[dict[str, Any]]]:
    """Infer one transition with bounded, fully retained format correction."""

    if maximum_format_repairs < 0:
        raise ValueError("maximum_format_repairs must be non-negative")
    samples = tuple(int(item) for item in sample_indices)
    allowed = (
        tuple(int(item) for item in allowed_sample_positions)
        if allowed_sample_positions is not None
        else None
    )
    attempts: list[dict[str, Any]] = []
    current_prompt = prompt
    for attempt_index in range(maximum_format_repairs + 1):
        raw_text = infer(current_prompt)
        try:
            parsed = parse_temporal_boundary_proposal(
                raw_text,
                episode_frames=episode_frames,
                sample_indices=samples,
                allowed_sample_positions=allowed,
            )
        except ValueError as exc:
            error = str(exc)
            attempt = {
                "attempt": attempt_index,
                "prompt_kind": "initial" if attempt_index == 0 else "format_repair",
                "valid": False,
                "parse_error": error,
                "raw_text": raw_text,
            }
            attempts.append(attempt)
            if record_attempt is not None:
                record_attempt(attempt)
            if attempt_index == maximum_format_repairs:
                return None, attempts
            current_prompt = temporal_boundary_format_repair_prompt(
                prompt,
                error,
                repair_number=attempt_index + 1,
            )
            continue
        attempt = {
            "attempt": attempt_index,
            "prompt_kind": "initial" if attempt_index == 0 else "format_repair",
            "valid": True,
            "parse_error": None,
            "raw_text": raw_text,
        }
        attempts.append(attempt)
        if record_attempt is not None:
            record_attempt(attempt)
        return parsed, attempts
    raise AssertionError("bounded temporal boundary inference loop did not return")


def combine_temporal_boundaries(
    rows: Iterable[TemporalBoundaryProposal],
) -> TemporalProposal:
    """Combine independently grounded transitions without modifying their values."""

    proposals = list(rows)
    if not proposals:
        raise ValueError("at least one independently grounded transition is required")
    boundaries = tuple(row.frame_index for row in proposals)
    if boundaries != tuple(sorted(set(boundaries))):
        raise ValueError("independently grounded boundaries must be strictly increasing")
    return TemporalProposal(
        boundaries=boundaries,
        confidences=tuple(row.confidence for row in proposals),
        evidence=tuple(row.evidence for row in proposals),
        raw_text="\n".join(row.raw_text for row in proposals),
    )


def temporal_format_repair_prompt(
    original_prompt: str,
    _invalid_text: str,
    error: str,
    *,
    expected_boundaries: int,
    repair_number: int = 1,
    boundary_key: str = "boundary_frame_indices",
) -> str:
    """Build a deterministic corrective prompt without changing label gates."""

    emphasis = (
        "Do not repeat a transition list or any prior answer."
        if repair_number == 1
        else "Use compact single-line JSON; keep every evidence phrase under six words."
    )
    targeted = ""
    if "evidence" in error:
        targeted = (
            " CRITICAL: visible_evidence must be a JSON ARRAY in square brackets, never "
            f"one string. Put exactly {expected_boundaries} separately quoted strings inside it."
        )
    elif "confidences" in error:
        targeted = (
            " CRITICAL: boundary_confidences must be a JSON ARRAY in square brackets with "
            f"exactly {expected_boundaries} numeric values."
        )
    elif "boundar" in error:
        targeted = (
            f" CRITICAL: {boundary_key} must be a JSON ARRAY in square brackets with "
            f"exactly {expected_boundaries} strictly increasing integers from the allowed range."
        )
    return (
        f"{original_prompt}\n\n"
        "FORMAT_CORRECTION: Your previous answer failed strict validation. "
        f"Validation error: {error}. Return exactly one JSON object and no other text. "
        f"Each of the three lists must contain exactly {expected_boundaries} entries. "
        f"Use the required {boundary_key} key and strictly increasing allowed values. "
        "Preserve your visual "
        f"judgment; this retry corrects structure only. {emphasis}{targeted}\n"
        "FINAL_JSON_ONLY:"
    )


def infer_temporal_proposal_with_repair(
    infer: Callable[[str], str],
    prompt: str,
    *,
    expected_boundaries: int,
    episode_frames: int,
    allowed_boundary_indices: Iterable[int] | None = None,
    sample_indices: Iterable[int] | None = None,
    maximum_format_repairs: int,
    record_attempt: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[TemporalProposal | None, list[dict[str, Any]]]:
    """Infer a proposal while retaining every valid or malformed response.

    Format correction is bounded and never modifies an answer locally.
    Exhausted retries return ``None`` so callers can reject the episode without
    terminating unrelated adjudications.
    """

    if maximum_format_repairs < 0:
        raise ValueError("maximum_format_repairs must be non-negative")
    attempts: list[dict[str, Any]] = []
    frozen_sample_indices = (
        tuple(int(item) for item in sample_indices) if sample_indices is not None else None
    )
    boundary_key = (
        "boundary_sample_positions"
        if frozen_sample_indices is not None
        else "boundary_frame_indices"
    )
    current_prompt = prompt
    for attempt_index in range(maximum_format_repairs + 1):
        raw_text = infer(current_prompt)
        try:
            parsed = parse_temporal_proposal(
                raw_text,
                expected_boundaries=expected_boundaries,
                episode_frames=episode_frames,
                allowed_boundary_indices=allowed_boundary_indices,
                sample_indices=frozen_sample_indices,
            )
        except ValueError as exc:
            error = str(exc)
            attempt = {
                "attempt": attempt_index,
                "prompt_kind": "initial" if attempt_index == 0 else "format_repair",
                "valid": False,
                "parse_error": error,
                "raw_text": raw_text,
            }
            attempts.append(attempt)
            if record_attempt is not None:
                record_attempt(attempt)
            if attempt_index == maximum_format_repairs:
                return None, attempts
            current_prompt = temporal_format_repair_prompt(
                prompt,
                raw_text,
                error,
                expected_boundaries=expected_boundaries,
                repair_number=attempt_index + 1,
                boundary_key=boundary_key,
            )
            continue
        attempt = {
            "attempt": attempt_index,
            "prompt_kind": "initial" if attempt_index == 0 else "format_repair",
            "valid": True,
            "parse_error": None,
            "raw_text": raw_text,
        }
        attempts.append(attempt)
        if record_attempt is not None:
            record_attempt(attempt)
        return parsed, attempts
    raise AssertionError("bounded temporal inference loop did not return")


def contact_sheet_indices(length: int, count: int, variant: int) -> np.ndarray:
    """Return deterministic, staggered local frame indices for one episode."""

    if length < 2 or count < 3:
        raise ValueError("contact-sheet sampling requires length >= 2 and count >= 3")
    if variant not in {0, 1, 2}:
        raise ValueError("the frozen consensus uses exactly three sampling variants")
    count = min(count, length)
    # Each variant observes the whole trajectory while moving interior samples
    # within their temporal bins.  Endpoints remain fixed for task context.
    positions = np.linspace(0.0, 1.0, count)
    if count > 2:
        bin_width = 1.0 / (count - 1)
        shift = (-0.25, 0.0, 0.25)[variant] * bin_width
        positions[1:-1] = np.clip(positions[1:-1] + shift, 0.0, 1.0)
    indices = np.rint(positions * (length - 1)).astype(np.int64)
    indices[0], indices[-1] = 0, length - 1
    if len(np.unique(indices)) != len(indices):
        indices = np.linspace(0, length - 1, count, dtype=np.int64)
    return indices


def constrained_sample_positions(
    sample_indices: Iterable[int],
    *,
    previous_frame: int,
    episode_frames: int,
    remaining_phases: int,
    minimum_phase_frames: int,
) -> tuple[int, ...]:
    """Return positions satisfying frozen sequential phase-duration constraints."""

    if remaining_phases < 1 or minimum_phase_frames < 1:
        raise ValueError("remaining_phases and minimum_phase_frames must be positive")
    return tuple(
        position
        for position, frame in enumerate(int(item) for item in sample_indices)
        if frame - previous_frame >= minimum_phase_frames
        and episode_frames - frame >= remaining_phases * minimum_phase_frames
    )


def _json_objects(text: str) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def parse_temporal_proposal(
    text: str,
    *,
    expected_boundaries: int,
    episode_frames: int,
    allowed_boundary_indices: Iterable[int] | None = None,
    sample_indices: Iterable[int] | None = None,
) -> TemporalProposal:
    """Parse the final strict JSON object emitted by the temporal VLM."""

    objects = list(_json_objects(text))
    if not objects:
        raise ValueError("temporal VLM output contains no JSON object")
    value = objects[-1]
    samples = tuple(int(item) for item in sample_indices) if sample_indices is not None else None
    boundary_key = "boundary_sample_positions" if samples is not None else "boundary_frame_indices"
    raw_boundaries = value.get(boundary_key)
    raw_confidences = value.get("boundary_confidences")
    raw_evidence = value.get("visible_evidence")
    if not isinstance(raw_boundaries, list) or len(raw_boundaries) != expected_boundaries:
        raise ValueError("temporal VLM returned the wrong number of boundaries")
    if not isinstance(raw_confidences, list) or len(raw_confidences) != expected_boundaries:
        raise ValueError("temporal VLM returned the wrong number of confidences")
    if not isinstance(raw_evidence, list) or len(raw_evidence) != expected_boundaries:
        raise ValueError("temporal VLM returned the wrong number of evidence statements")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in raw_boundaries):
        raise ValueError(f"{boundary_key} must contain JSON integers")
    raw_boundary_tuple = tuple(int(item) for item in raw_boundaries)
    if raw_boundary_tuple != tuple(sorted(set(raw_boundary_tuple))):
        raise ValueError("temporal VLM boundaries must be strictly increasing")
    if samples is not None:
        if any(item < 0 or item >= len(samples) for item in raw_boundary_tuple):
            raise ValueError("temporal VLM sample position lies outside the contact sheet")
        boundaries = tuple(samples[item] for item in raw_boundary_tuple)
    else:
        boundaries = raw_boundary_tuple
    if any(item <= 0 or item >= episode_frames for item in boundaries):
        raise ValueError("temporal VLM boundary lies outside the episode")
    if allowed_boundary_indices is not None:
        allowed = {int(item) for item in allowed_boundary_indices}
        if any(item not in allowed for item in boundaries):
            raise ValueError("temporal VLM boundary is not a sampled frame label")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in raw_confidences):
        raise ValueError("boundary_confidences must be numeric")
    confidences = tuple(float(item) for item in raw_confidences)
    if any(not 0.0 <= item <= 1.0 for item in confidences):
        raise ValueError("boundary confidence lies outside [0, 1]")
    if any(not isinstance(item, str) or not item.strip() for item in raw_evidence):
        raise ValueError("visible_evidence must contain non-empty strings")
    return TemporalProposal(
        boundaries=boundaries,
        confidences=confidences,
        evidence=tuple(str(item).strip() for item in raw_evidence),
        raw_text=text,
    )


def temporal_consensus(
    proposals: Iterable[TemporalProposal],
    *,
    episode_frames: int,
    skills: int,
    minimum_phase_frames: int,
    minimum_confidence: float,
    maximum_spread_fraction: float,
) -> dict[str, Any]:
    """Accept only high-confidence, three-view boundary consensus."""

    rows = list(proposals)
    expected = skills - 1
    reasons: list[str] = []
    if len(rows) != 3:
        reasons.append("requires_exactly_three_sampling_variants")
    if any(len(row.boundaries) != expected for row in rows):
        reasons.append("wrong_boundary_count")
    if reasons:
        return {"accepted": False, "reasons": reasons}

    matrix = np.asarray([row.boundaries for row in rows], dtype=np.int64)
    confidence = np.asarray([row.confidences for row in rows], dtype=np.float64)
    consensus = np.rint(np.median(matrix, axis=0)).astype(np.int64)
    spread = np.ptp(matrix, axis=0)
    maximum_spread_frames = max(1, int(np.floor(maximum_spread_fraction * episode_frames)))
    if float(confidence.min()) < minimum_confidence:
        reasons.append("confidence_below_frozen_minimum")
    if np.any(spread > maximum_spread_frames):
        reasons.append("sampling_variants_disagree")
    phases = np.diff(np.concatenate([[0], consensus, [episode_frames]]))
    if np.any(phases < minimum_phase_frames):
        reasons.append("consensus_creates_undersized_phase")
    if tuple(consensus.tolist()) != tuple(sorted(set(consensus.tolist()))):
        reasons.append("consensus_is_not_strictly_monotonic")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "boundaries": consensus.tolist(),
        "variant_boundaries": matrix.tolist(),
        "boundary_spread_frames": spread.tolist(),
        "maximum_spread_frames": maximum_spread_frames,
        "minimum_observed_confidence": float(confidence.min()),
        "mean_observed_confidence": float(confidence.mean()),
        "visible_evidence": [list(row.evidence) for row in rows],
    }


__all__ = [
    "TemporalBoundaryProposal",
    "TemporalProposal",
    "combine_temporal_boundaries",
    "constrained_sample_positions",
    "contact_sheet_indices",
    "infer_temporal_boundary_with_repair",
    "infer_temporal_proposal_with_repair",
    "parse_temporal_boundary_proposal",
    "parse_temporal_proposal",
    "temporal_boundary_format_repair_prompt",
    "temporal_format_repair_prompt",
    "temporal_consensus",
]
