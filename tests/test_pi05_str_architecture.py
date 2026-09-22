"""PI0.5 STR routes active failures without changing the base checkpoint."""

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


class TestPI05STRArchitecture(unittest.TestCase):
    def test_only_str_routes_an_active_failure_to_recovery(self) -> None:
        manifest = json.loads(
            (ROOT / "benchmark/methods/pi05_str.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["features"]["recovery"], True)
        self.assertEqual(manifest["features"]["retrained"], False)
        programs = load_method_programs(ROOT / manifest["language_program"])
        env = types.SimpleNamespace(scene={"robot": _Asset(), "box": _Asset()})
        st = HumanoidArenaMethodRuntime(
            programs, method_id="pi05_st", task_id="pick_and_place_box",
            control_dt_s=0.02, seed=7,
        )
        str_runtime = HumanoidArenaMethodRuntime(
            programs, method_id="pi05_str", task_id="pick_and_place_box",
            control_dt_s=0.02, seed=7,
        )
        st.reset(env)
        str_runtime.reset(env)
        kwargs = {
            "recovery_scenario_id": "box-missed-grasp-retry",
            "recovery_active": True,
        }
        self.assertNotEqual(st.instruction(env, **kwargs).route, "transition_recovery")
        recovery = str_runtime.instruction(env, **kwargs)
        self.assertEqual(recovery.route, "transition_recovery")
        self.assertIn("re-align", recovery.instruction)
