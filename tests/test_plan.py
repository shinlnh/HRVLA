import unittest
from pathlib import Path

from hrvla_bench.plan import build_plan, load_json, validate_suite


SUITE = Path(__file__).resolve().parents[1] / "benchmark/suites/hrvla_recovery_v0.json"


class PlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.suite = load_json(SUITE)

    def test_suite_is_valid_and_drafts_are_excluded(self) -> None:
        validate_suite(self.suite)
        plan = build_plan(self.suite, ["b", "a"], rollouts_override=2)
        self.assertEqual(plan["methods"], ["a", "b"])
        self.assertEqual(plan["episodes"], [])

    def test_diagnostic_plan_is_deterministic(self) -> None:
        first = build_plan(
            self.suite, ["a", "b"], include_draft=True, rollouts_override=2
        )
        second = build_plan(
            self.suite, ["b", "a"], include_draft=True, rollouts_override=2
        )
        self.assertEqual(first["plan_sha256"], second["plan_sha256"])
        self.assertEqual(len(first["episodes"]), 96)
        self.assertTrue(all(item["admission_status"] == "draft" for item in first["episodes"]))


if __name__ == "__main__":
    unittest.main()
