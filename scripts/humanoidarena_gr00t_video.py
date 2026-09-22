#!/usr/bin/env python3
"""Install the packed-video offset shim used by HumanoidArena GR00T views.

LeRobot v3 stores many episodes in one MP4.  The lightweight GR00T dataset view
uses one symlink per episode and records the episode's exact frame offset in
``episodes.jsonl``.  Isaac-GR00T's v2 loader can then be reused unchanged apart
from adding that offset before frame decode.  Ordinary v2 datasets have no
offset field and retain byte-for-byte-equivalent behavior.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def install_packed_video_offset_patch() -> None:
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    if getattr(LeRobotEpisodeLoader, "_hrvla_packed_video_patch", False):
        return
    original = LeRobotEpisodeLoader._load_video_data

    def load_video_data(self: Any, episode_index: int, indices: Any):
        metadata = self.episodes_metadata[episode_index]
        offset = int(metadata.get("video_frame_offset", 0))
        if offset:
            indices = np.asarray(indices, dtype=np.int64) + offset
        return original(self, episode_index, indices)

    LeRobotEpisodeLoader._load_video_data = load_video_data
    LeRobotEpisodeLoader._hrvla_packed_video_patch = True
