"""Modality contract for the Isaac Lab Arena Unitree G1 dataset.

The public Arena dataset exposes direct 43-DoF joint targets rather than SONIC
motion tokens.  It is therefore used to post-train the VLA action adapter and
to measure in-domain action prediction; the existing SONIC adapter remains the
deployment boundary.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


JOINT_GROUPS = [
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "left_hand",
    "right_arm",
    "right_hand",
]

ARENA_G1_SUBTASK_CONFIG = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=JOINT_GROUPS),
    "action": ModalityConfig(
        delta_indices=list(range(40)),
        modality_keys=JOINT_GROUPS,
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in JOINT_GROUPS
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(ARENA_G1_SUBTASK_CONFIG, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
