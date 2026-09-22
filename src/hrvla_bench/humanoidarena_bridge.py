"""Contracts for adapting released HumanoidArena data to Isaac-GR00T.

HumanoidArena's ``sonic_refpose_v3_1`` datasets expose a flat 64-dimensional
observation and a flat 40-dimensional reference-pose action.  Isaac-GR00T
expects named modality groups.  This module owns the one lossless mapping used
by dataset preparation, training, inference, and contract tests.

The bridge does not translate a direct joint action into a SONIC latent.  Its
40-D output is already the semantic reference-pose contract consumed by
HumanoidArena's released ``SonicActionProvider``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any, Iterable, Mapping, Sequence


STATE_DIM = 64
ACTION_DIM = 40
ACTION_HORIZON = 40
VIDEO_KEY = "front"
LANGUAGE_KEY = "annotation.human.task_description"
VLA_SCHEMA = "unitree_g1_gmt_refpose_v3_1"
VLA_SCHEMA_VERSION = "3.1"
ACTION_LAYOUT = "root_xy_delta_z_rot6d_joints29_hands2"
ROTATION_6D_LAYOUT = "row"


@dataclass(frozen=True)
class VectorField:
    """One named, half-open slice in a flat state or action vector."""

    name: str
    start: int
    end: int

    @property
    def width(self) -> int:
        return self.end - self.start


STATE_FIELDS = (
    VectorField("root_heading_canonical_rot6d", 0, 6),
    VectorField("joint_pos", 6, 35),
    VectorField("joint_vel", 35, 64),
)

ACTION_FIELDS = (
    VectorField("root_ref_base_local_xy_delta", 0, 2),
    VectorField("root_z", 2, 3),
    VectorField("root_ref_rot6d", 3, 9),
    VectorField("joint_pos", 9, 38),
    VectorField("hand_binary", 38, 40),
)


def _validate_fields(fields: Sequence[VectorField], expected_dim: int) -> None:
    cursor = 0
    names: set[str] = set()
    for field in fields:
        if field.name in names:
            raise ValueError(f"duplicate vector field: {field.name}")
        if field.start != cursor or field.end <= field.start:
            raise ValueError(f"non-contiguous vector field: {field}")
        names.add(field.name)
        cursor = field.end
    if cursor != expected_dim:
        raise ValueError(f"field layout ends at {cursor}, expected {expected_dim}")


_validate_fields(STATE_FIELDS, STATE_DIM)
_validate_fields(ACTION_FIELDS, ACTION_DIM)


def split_vector(
    values: Sequence[Any], fields: Sequence[VectorField], expected_dim: int
) -> dict[str, list[Any]]:
    """Split a flat vector into named groups without changing values or order."""

    if len(values) != expected_dim:
        raise ValueError(f"expected vector width {expected_dim}, got {len(values)}")
    return {field.name: list(values[field.start : field.end]) for field in fields}


def split_state(values: Sequence[Any]) -> dict[str, list[Any]]:
    return split_vector(values, STATE_FIELDS, STATE_DIM)


def split_action(values: Sequence[Any]) -> dict[str, list[Any]]:
    return split_vector(values, ACTION_FIELDS, ACTION_DIM)


def flatten_groups(
    groups: Mapping[str, Sequence[Any]], fields: Sequence[VectorField], expected_dim: int
) -> list[Any]:
    """Reassemble named groups, rejecting missing, extra, or malformed fields."""

    expected_names = {field.name for field in fields}
    actual_names = set(groups)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(f"group mismatch: missing={missing}, extra={extra}")
    output: list[Any] = []
    for field in fields:
        values = list(groups[field.name])
        if len(values) != field.width:
            raise ValueError(
                f"{field.name} has width {len(values)}, expected {field.width}"
            )
        output.extend(values)
    if len(output) != expected_dim:
        raise AssertionError("validated groups did not reconstruct the declared dimension")
    return output


def flatten_action(groups: Mapping[str, Sequence[Any]]) -> list[Any]:
    return flatten_groups(groups, ACTION_FIELDS, ACTION_DIM)


def deterministic_split(
    episode_ids: Iterable[int], heldout_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Return a deterministic per-task train/held-out split."""

    ordered = sorted(set(int(value) for value in episode_ids))
    if not ordered:
        raise ValueError("no episode IDs")
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be between zero and one")
    shuffled = ordered.copy()
    random.Random(seed).shuffle(shuffled)
    heldout_count = max(1, round(len(shuffled) * heldout_fraction))
    heldout = sorted(shuffled[:heldout_count])
    train = sorted(shuffled[heldout_count:])
    return train, heldout


def validate_release_info(info: Mapping[str, Any]) -> None:
    """Reject a dataset whose observation/action semantics differ from the bridge."""

    features = info.get("features", {})
    expected_features = {
        "observation.images.front": (480, 640, 3),
        "observation.state": (STATE_DIM,),
        "action": (ACTION_DIM,),
    }
    for key, shape in expected_features.items():
        feature = features.get(key)
        actual_shape = tuple(feature.get("shape", ())) if isinstance(feature, Mapping) else ()
        if actual_shape != shape:
            raise ValueError(f"{key} shape is {actual_shape}, expected {shape}")

    protocol = info.get("vla_protocol", {})
    expected_protocol = {
        "schema": VLA_SCHEMA,
        "version": VLA_SCHEMA_VERSION,
        "action_dim": ACTION_DIM,
        "action_layout": ACTION_LAYOUT,
        "rotation_6d_layout": ROTATION_6D_LAYOUT,
        "action_semantics": "reference_pose_not_robot_current_residual",
        "root_xy_delta_frame": "current_reference_base_frame",
        "root_rotation_frame": "episode_reference_frame",
        "backend_source": "sonic",
    }
    mismatches = {
        key: {"actual": protocol.get(key), "expected": expected}
        for key, expected in expected_protocol.items()
        if protocol.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"incompatible HumanoidArena VLA protocol: {mismatches}")


def modality_metadata() -> dict[str, Any]:
    """Return LeRobot-v2 modality slices matching the GR00T bridge config."""

    return {
        "state": {
            field.name: {
                "start": field.start,
                "end": field.end,
                "original_key": "observation.state",
            }
            for field in STATE_FIELDS
        },
        "action": {
            field.name: {
                "start": field.start,
                "end": field.end,
                "original_key": "action",
            }
            for field in ACTION_FIELDS
        },
        "video": {VIDEO_KEY: {"original_key": "observation.images.front"}},
        "annotation": {
            "human.task_description": {"original_key": "task_index"}
        },
    }


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()

