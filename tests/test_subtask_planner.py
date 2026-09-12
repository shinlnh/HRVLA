from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hrvla_subtask.model import Candidate, ExecutionMemory  # noqa: E402
from hrvla_subtask.planner import PlannerConfig, WorldModelGuidedPlanner  # noqa: E402
from hrvla_subtask.simulator import load_suite, run_episode  # noqa: E402


class FixedBackend:
    def __init__(self, candidates: list[Candidate]) -> None:
        self.candidates = candidates
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def generated_tokens(self) -> int:
        return 0

    def propose(self, task, state, memory, count, seed):
        self._calls += 1
        return [self.candidates[index % len(self.candidates)] for index in range(count)]


class PlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.task = load_suite(REPO_ROOT / "benchmark" / "subtask_suite_v1.json")[0]

    def test_best_of_n_rejects_inapplicable_high_confidence_candidate(self) -> None:
        backend = FixedBackend(
            [
                Candidate("return_home", 0.99, "premature"),
                Candidate("grasp_apple", 0.70, "grounded"),
            ]
        )
        planner = WorldModelGuidedPlanner(
            backend,
            PlannerConfig(method="best_of_n", branching_factor=2),
        )
        decision = planner.decide(self.task, self.task.initial_state, ExecutionMemory())
        self.assertEqual(decision.selected.skill_id, "grasp_apple")
        self.assertFalse(decision.safety_fallback)

    def test_adaptive_router_invokes_search_when_direct_proposal_is_uncertain(self) -> None:
        backend = FixedBackend([Candidate("grasp_apple", 0.20)])
        planner = WorldModelGuidedPlanner(
            backend,
            PlannerConfig(
                method="adaptive_ttc",
                branching_factor=1,
                beam_width=1,
                search_depth=2,
                confidence_threshold=0.7,
            ),
        )
        decision = planner.decide(self.task, self.task.initial_state, ExecutionMemory())
        self.assertTrue(decision.route.startswith("ttc"))
        self.assertGreaterEqual(decision.search_nodes, 1)

    def test_full_ttc_is_reported_as_a_ttc_route(self) -> None:
        backend = FixedBackend([Candidate("grasp_apple", 0.90)])
        planner = WorldModelGuidedPlanner(
            backend,
            PlannerConfig(method="ttc", branching_factor=1, beam_width=1, search_depth=1),
        )
        decision = planner.decide(self.task, self.task.initial_state, ExecutionMemory())
        self.assertTrue(decision.route.startswith("ttc"))

    def test_closed_loop_recovers_declared_drop(self) -> None:
        result = run_episode(
            self.task,
            method="ttc",
            seed=7,
            proposal_error_rate=0.0,
            planner_overrides={"branching_factor": 2, "beam_width": 2, "search_depth": 2},
        )
        self.assertTrue(result.success)
        self.assertEqual(result.injected_failures, 1)
        self.assertEqual(result.recovered_failures, 1)
        self.assertLessEqual(result.recovered_failures, result.injected_failures)
        self.assertTrue(any(item["outcome"] == "injected_failure" for item in result.trace))

    def test_stochastic_failures_do_not_inflate_injected_recovery(self) -> None:
        result = run_episode(
            self.task,
            method="plan_once",
            seed=3,
            proposal_error_rate=0.0,
        )
        self.assertLessEqual(result.recovered_failures, result.injected_failures)


if __name__ == "__main__":
    unittest.main()
