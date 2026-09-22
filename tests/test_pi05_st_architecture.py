"""PI0.5 ST must change only instruction selection, not the policy family."""

import json
from pathlib import Path
import types
import unittest

import torch

from hrvla_bench.humanoidarena_method_runtime import (
    HumanoidArenaMethodRuntime,
    load_method_programs,
)


ROOT = Path(__file__).resolve().parents[1]


class _Asset:
    def __init__(self) -> None:
        pose = torch.zeros((1, 7))
        pose[:, 3] = 1.0
        self.data = types.SimpleNamespace(
            root_link_pose_w=pose,
            root_state_w=torch.cat((pose, torch.zeros((1, 6))), dim=1),
            root_vel_w=torch.zeros((1, 6)),
        )


class TestPI05STArchitecture(unittest.TestCase):
    def test_st_selects_subtask_without_changing_checkpoint(self) -> None:
        manifest = json.loads(
            (ROOT / "benchmark/methods/pi05_st.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["parent"], "pi05_sonic")
        self.assertEqual(manifest["features"], {
            "subtask": True,
            "recovery": False,
            "retrained": False,
        })
        programs = load_method_programs(ROOT / manifest["language_program"])
        env = types.SimpleNamespace(scene={"robot": _Asset(), "box": _Asset()})
        runtime = HumanoidArenaMethodRuntime(
            programs,
            method_id="pi05_st",
            task_id="pick_and_place_box",
            control_dt_s=0.02,
            seed=7,
        )
        runtime.reset(env)
        first = runtime.instruction(env)
        self.assertEqual(first.selected_skill_id, "approach_and_grasp_box")
        close = [0.0] * 38 + [1.0, 1.0]
        second = runtime.instruction(env, semantic_action=close)
        self.assertTrue(second.transition_triggered)
        self.assertEqual(second.selected_skill_id, "lift_and_transport_box")
