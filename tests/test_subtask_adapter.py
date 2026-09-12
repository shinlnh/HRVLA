from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hrvla_subtask.adapter import Gr00tSubtaskAdapter  # noqa: E402
from hrvla_subtask.backends import HeuristicProposalBackend  # noqa: E402
from hrvla_subtask.planner import PlannerConfig, WorldModelGuidedPlanner  # noqa: E402
from hrvla_subtask.simulator import load_suite  # noqa: E402


class RecordingPolicy:
    def __init__(self) -> None:
        self.observation = None

    def get_action(self, observation):
        self.observation = observation
        return {"action.motion_token": "unchanged"}, {"ok": True}


class AdapterTest(unittest.TestCase):
    def test_only_language_prompt_is_replaced(self) -> None:
        task = load_suite(REPO_ROOT / "benchmark" / "subtask_suite_v1.json")[0]
        policy = RecordingPolicy()
        planner = WorldModelGuidedPlanner(
            HeuristicProposalBackend(error_rate=0.0), PlannerConfig(method="plan_once")
        )
        original = {
            "video": {"ego_view": [[[1, 2, 3]]]},
            "state": {"joint": [[0.0]]},
            "language": {"annotation.human.task_description": [[task.goal_instruction]]},
            "symbolic_state": task.initial_state,
        }
        untouched = deepcopy(original)
        adapter = Gr00tSubtaskAdapter(
            policy,
            planner,
            task,
            state_extractor=lambda observation: observation["symbolic_state"],
        )
        result = adapter.get_action(original)
        self.assertEqual(result[0]["action.motion_token"], "unchanged")
        self.assertEqual(original, untouched)
        self.assertEqual(policy.observation["video"], original["video"])
        self.assertEqual(
            policy.observation["language"]["annotation.human.task_description"],
            [["Grasp the red apple from the table."]],
        )


if __name__ == "__main__":
    unittest.main()
