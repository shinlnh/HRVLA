from __future__ import annotations

import numpy as np
import pytest

from hrvla_bench.subtask_relabel import segment_subtasks
from hrvla_bench.subtask_video_adjudication import (
    TemporalProposal,
    contact_sheet_indices,
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
