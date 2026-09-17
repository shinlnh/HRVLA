from __future__ import annotations

import numpy as np
import pytest

from hrvla_bench.subtask_relabel import segment_subtasks
from hrvla_bench.subtask_video_adjudication import (
    TemporalBoundaryProposal,
    TemporalProposal,
    combine_temporal_boundaries,
    contact_sheet_indices,
    infer_temporal_boundary_with_repair,
    infer_temporal_proposal_with_repair,
    parse_temporal_boundary_proposal,
    parse_temporal_proposal,
    temporal_consensus,
)


def _proposal(boundaries: tuple[int, int], confidence: float = 0.9) -> TemporalProposal:
    return TemporalProposal(
        boundaries=boundaries,
        confidences=(confidence, confidence),
        evidence=("wrist reaches handle", "door is visibly open"),
        raw_text="{}",
    )


def test_contact_sheet_variants_are_ordered_and_staggered() -> None:
    variants = [contact_sheet_indices(600, 24, variant) for variant in range(3)]
    assert all(row[0] == 0 and row[-1] == 599 for row in variants)
    assert all(np.all(np.diff(row) > 0) for row in variants)
    assert not np.array_equal(variants[0], variants[1])


def test_parse_temporal_proposal_uses_final_strict_json() -> None:
    text = (
        '<think>{"irrelevant": true}</think>\n'
        '{"boundary_frame_indices":[81,161],"boundary_confidences":[0.91,0.88],'
        '"visible_evidence":["handle contact","door opened"]}'
    )
    parsed = parse_temporal_proposal(text, expected_boundaries=2, episode_frames=240)
    assert parsed.boundaries == (81, 161)
    assert parsed.confidences == (0.91, 0.88)


def test_parse_sample_positions_maps_exactly_to_contact_sheet_frames() -> None:
    text = (
        '{"boundary_sample_positions":[1,3],"boundary_confidences":[0.91,0.88],'
        '"visible_evidence":["handle contact","door opened"]}'
    )
    parsed = parse_temporal_proposal(
        text,
        expected_boundaries=2,
        episode_frames=240,
        sample_indices=[0, 80, 120, 160, 239],
    )
    assert parsed.boundaries == (80, 160)


def test_independent_transition_grounding_combines_without_value_changes() -> None:
    samples = [0, 80, 120, 160, 239]
    first = parse_temporal_boundary_proposal(
        '{"boundary_sample_position":1,"boundary_confidence":0.91,'
        '"visible_evidence":"gripper closes"}',
        episode_frames=240,
        sample_indices=samples,
    )
    second = parse_temporal_boundary_proposal(
        '{"boundary_sample_position":3,"boundary_confidence":0.88,'
        '"visible_evidence":"door starts opening"}',
        episode_frames=240,
        sample_indices=samples,
    )
    combined = combine_temporal_boundaries([first, second])
    assert combined.boundaries == (80, 160)
    assert combined.confidences == (0.91, 0.88)
    with pytest.raises(ValueError, match="strictly increasing"):
        combine_temporal_boundaries([second, first])


def test_independent_transition_format_repair_is_bounded_and_retained() -> None:
    outputs = iter(
        [
            '{"boundary_sample_position":[1],"boundary_confidence":0.9,'
            '"visible_evidence":"contact"}',
            '{"boundary_sample_position":1,"boundary_confidence":0.9,'
            '"visible_evidence":"contact"}',
        ]
    )
    prompts: list[str] = []
    retained: list[dict] = []

    def infer(prompt: str) -> str:
        prompts.append(prompt)
        return next(outputs)

    proposal, attempts = infer_temporal_boundary_with_repair(
        infer,
        "one transition",
        episode_frames=240,
        sample_indices=[0, 80, 160, 239],
        maximum_format_repairs=2,
        record_attempt=retained.append,
    )
    assert proposal is not None and proposal.frame_index == 80
    assert [row["valid"] for row in attempts] == [False, True]
    assert retained == attempts
    assert "three scalar values" not in prompts[1]
    assert "Do not return lists" in prompts[1]


def test_combine_rejects_duplicate_independent_boundaries() -> None:
    row = TemporalBoundaryProposal(80, 1, 0.9, "contact", "{}")
    with pytest.raises(ValueError, match="strictly increasing"):
        combine_temporal_boundaries([row, row])


def test_format_repair_retains_invalid_attempt_then_accepts_strict_json() -> None:
    invalid = (
        '{"boundary_frame_indices":[80],"boundary_confidences":[0.9],'
        '"visible_evidence":["contact"]}'
    )
    outputs = iter(
        [
            invalid,
            '{"boundary_frame_indices":[80,160],"boundary_confidences":[0.9,0.8],'
            '"visible_evidence":["contact","opened"]}',
        ]
    )
    prompts: list[str] = []
    retained: list[dict] = []

    def infer(prompt: str) -> str:
        prompts.append(prompt)
        return next(outputs)

    proposal, attempts = infer_temporal_proposal_with_repair(
        infer,
        "SAMPLED_LOCAL_FRAMES: [0, 80, 160, 239]",
        expected_boundaries=2,
        episode_frames=240,
        allowed_boundary_indices=[0, 80, 160, 239],
        maximum_format_repairs=2,
        record_attempt=retained.append,
    )
    assert proposal is not None and proposal.boundaries == (80, 160)
    assert [row["valid"] for row in attempts] == [False, True]
    assert retained == attempts
    assert "FORMAT_CORRECTION" in prompts[1]
    assert "exactly 2 entries" in prompts[1]
    assert "JSON ARRAY in square brackets" in prompts[1]
    assert invalid not in prompts[1]


def test_format_repair_is_bounded_and_does_not_invent_a_boundary() -> None:
    invalid = (
        '{"boundary_frame_indices":[81,160],"boundary_confidences":[0.9,0.8],'
        '"visible_evidence":["contact","opened"]}'
    )
    calls = 0

    def infer(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return invalid

    proposal, attempts = infer_temporal_proposal_with_repair(
        infer,
        "sampled labels",
        expected_boundaries=2,
        episode_frames=240,
        allowed_boundary_indices=[0, 80, 160, 239],
        maximum_format_repairs=2,
    )
    assert proposal is None
    assert calls == len(attempts) == 3
    assert all(not row["valid"] for row in attempts)
    assert all("not a sampled frame label" in row["parse_error"] for row in attempts)


def test_consensus_rejects_sampling_disagreement() -> None:
    report = temporal_consensus(
        [_proposal((80, 160)), _proposal((82, 161)), _proposal((120, 200))],
        episode_frames=240,
        skills=3,
        minimum_phase_frames=40,
        minimum_confidence=0.7,
        maximum_spread_fraction=0.08,
    )
    assert report["accepted"] is False
    assert "sampling_variants_disagree" in report["reasons"]


def test_accepted_consensus_replaces_only_an_explicit_fallback() -> None:
    report = temporal_consensus(
        [_proposal((78, 159)), _proposal((80, 160)), _proposal((82, 162))],
        episode_frames=240,
        skills=3,
        minimum_phase_frames=40,
        minimum_confidence=0.7,
        maximum_spread_fraction=0.08,
    )
    assert report["accepted"] is True
    state = np.zeros((240, 64), dtype=np.float32)
    action = np.zeros((240, 40), dtype=np.float32)
    segmented = segment_subtasks(
        "open_door",
        state,
        action,
        skill_count=3,
        boundary_override=report["boundaries"],
        boundary_override_audit=report,
    )
    assert segmented.audit["proxy"] == "cosmos-reason2-temporal-consensus"
    assert segmented.audit["fallback_used"] is False
    assert segmented.audit["fallback_avoided_by_adjudication"] is True


def test_unaccepted_adjudication_cannot_override_labels() -> None:
    state = np.zeros((240, 64), dtype=np.float32)
    action = np.zeros((240, 40), dtype=np.float32)
    with pytest.raises(ValueError, match="accepted adjudication"):
        segment_subtasks(
            "open_door",
            state,
            action,
            skill_count=3,
            boundary_override=[80, 160],
            boundary_override_audit={"accepted": False},
        )
