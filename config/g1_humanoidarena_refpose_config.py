"""Isaac-GR00T modality config for HumanoidArena's semantic 40-D SONIC input."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


STATE_KEYS = [
    "root_heading_canonical_rot6d",
    "joint_pos",
    "joint_vel",
]
ACTION_KEYS = [
    "root_ref_base_local_xy_delta",
    "root_z",
    "root_ref_rot6d",
    "joint_pos",
    "hand_binary",
]

HUMANOIDARENA_G1_REFPOSE_CONFIG = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["front"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=STATE_KEYS),
    "action": ModalityConfig(
        delta_indices=list(range(40)),
        modality_keys=ACTION_KEYS,
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in ACTION_KEYS
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    HUMANOIDARENA_G1_REFPOSE_CONFIG,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)

