from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hrvla_subtask.adapter import Gr00tRecoveryAdapter  # noqa: E402
from hrvla_subtask.recovery import (  # noqa: E402
    FailureContext,
    TransitionRecoveryCoordinator,
    load_recovery_protocol,
)
from hrvla_subtask.recovery_simulator import run_recovery_episode  # noqa: E402
from hrvla_subtask.simulator import load_suite  # noqa: E402


class RecordingPolicy:
    def __init__(self) -> None:
        self.observation = None

    def get_action(self, observation):
        self.observation = observation
        return {"action.motion_token": "unchanged"}


class RecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks = load_suite(REPO_ROOT / "benchmark" / "subtask_suite_v1.json")
        self.protocol = load_recovery_protocol(
            REPO_ROOT / "benchmark" / "recovery_protocol_v1.json", self.tasks
        )
        self.task = self.tasks[0]
        self.scenario = self.protocol[self.task.task_id]
        self.failure = self.task.failures[0]
        failed_state = set(self.task.initial_state)
        failed_state.discard("apple_on_table")
        failed_state.add("apple_on_floor")
        self.failed_state = frozenset(failed_state)

    def test_baton_selects_a_complete_handoff_contract(self) -> None:
        context = FailureContext(
            task_id=self.task.task_id,
            failure_label=self.failure.label,
            failed_skill_id=self.failure.skill_id,
            state_before=self.task.initial_state,
            observed_state=self.failed_state,
            attempt=1,
        )
        decision = TransitionRecoveryCoordinator("baton_str").decide(
            self.task, self.scenario, self.failure, context
        )
        primitive = self.scenario.primitive_map[decision.primitive_id]
        self.assertTrue(decision.contract_predicted)
        self.assertTrue(self.scenario.contract_satisfied(primitive.apply(self.failed_state)))

    def test_agentchord_proxy_uses_precompiled_forward_edge(self) -> None:
        context = FailureContext(
            task_id=self.task.task_id,
            failure_label=self.failure.label,
            failed_skill_id=self.failure.skill_id,
            state_before=self.task.initial_state,
            observed_state=self.failed_state,
            attempt=1,
        )
        decision = TransitionRecoveryCoordinator("agentchord").decide(
            self.task, self.scenario, self.failure, context
        )
        self.assertEqual(decision.primitive_id, "forward_regrasp_apple")
        self.assertFalse(decision.contract_predicted)

    def test_rekep_backtracking_does_not_teleport_world_state(self) -> None:
        result = run_recovery_episode(
            self.task,
            self.scenario,
            method="rekep",
            seed=5,
            disturbance_rate=0.0,
            proposal_error_rate=0.0,
        )
        restores = [row for row in result.trace if row["kind"] == "controller_backtrack"]
        self.assertTrue(restores)
        self.assertIn("apple_on_floor", restores[0]["state_after"])

    def test_recovery_adapter_only_changes_language(self) -> None:
        policy = RecordingPolicy()
        adapter = Gr00tRecoveryAdapter(policy)
        primitive = self.scenario.primitive_map["transition_regrasp_apple"]
        original = {
            "video": {"ego_view": [[[1, 2, 3]]]},
            "state": {"joint": [[0.0]]},
            "language": {"annotation.human.task_description": [["long task"]]},
        }
        untouched = deepcopy(original)
        adapter.get_action(original, primitive)
        self.assertEqual(original, untouched)
        self.assertEqual(policy.observation["video"], original["video"])
        self.assertEqual(
            policy.observation["language"]["annotation.human.task_description"],
            [[primitive.instruction]],
        )


if __name__ == "__main__":
    unittest.main()
