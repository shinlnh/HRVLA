import unittest

from hrvla_bench.score import score_records, validate_record


def episode(
    method: str,
    protocol: str,
    success: bool,
    scenario: str,
    rollout_seed: int,
    *,
    detected: bool | None = None,
) -> dict:
    record = {
        "schema_version": "1.0",
        "run_id": "synthetic-test-run",
        "episode_id": f"{method}-{protocol}-{scenario}-{rollout_seed}",
        "method_id": method,
        "evaluation_track": "end_to_end_recovery",
        "suite_id": "synthetic-suite",
        "suite_sha256": "a" * 64,
        "plan_sha256": "b" * 64,
        "task_id": "synthetic-task",
        "scenario_id": scenario,
        "protocol": protocol,
        "training_seed": 0,
        "rollout_seed": rollout_seed,
        "initial_snapshot_id": "initial-0",
        "policy_checkpoint_id": f"{method}-checkpoint",
        "controller_id": "fixed-controller",
        "simulator_revision": "fixed-simulator",
        "success": success,
        "termination": "success" if success else "timeout",
    }
    if detected is not None:
        record["detected"] = detected
    if protocol == "nominal":
        record["failure"] = None
    else:
        record["failure_snapshot_id"] = f"failure-{scenario}"
        record["failure"] = {
            "event_id": "after-stable-grasp",
            "event_boundary": "after stable grasp",
            "level": "L3",
            "humanoid_axes": ["H3"],
            "severity": "medium",
            "recoverable_oracle": True,
            "injector_id": "synthetic-injector",
            "injector_parameters": {},
        }
    return record


class ValidationTests(unittest.TestCase):
    def test_rejects_unverified_recovery_state(self) -> None:
        record = episode("a", "failure_start", True, "s0", 0)
        record["failure"]["recoverable_oracle"] = False
        with self.assertRaisesRegex(ValueError, "oracle-verified"):
            validate_record(record)

    def test_rejects_success_termination_mismatch(self) -> None:
        record = episode("a", "nominal", True, "n0", 0)
        record["termination"] = "timeout"
        with self.assertRaisesRegex(ValueError, "must agree"):
            validate_record(record)


class ScoringTests(unittest.TestCase):
    def test_scores_protocols_and_paired_difference(self) -> None:
        records = [
            episode("a", "nominal", True, "nominal", 0, detected=False),
            episode("b", "nominal", True, "nominal", 0, detected=False),
            episode("a", "failure_start", True, "s0", 0),
            episode("b", "failure_start", False, "s0", 0),
            episode("a", "failure_start", False, "s1", 1),
            episode("b", "failure_start", False, "s1", 1),
            episode("a", "online_failure", True, "s2", 2, detected=True),
            episode("b", "online_failure", False, "s2", 2, detected=False),
        ]
        report = score_records(records)
        methods = report["tracks"]["end_to_end_recovery"]["methods"]
        self.assertEqual(methods["a"]["nominal_success"]["rate"], 1.0)
        self.assertEqual(methods["a"]["failure_start"]["micro_rsr"]["rate"], 0.5)
        self.assertEqual(methods["a"]["online_failure"]["micro_rsr"]["rate"], 1.0)
        comparisons = {item["protocol"]: item for item in report["paired_comparisons"]}
        self.assertEqual(comparisons["failure_start"]["paired_episodes"], 2)
        self.assertAlmostEqual(
            comparisons["failure_start"]["success_rate_delta_a_minus_b"], 0.5
        )
        self.assertEqual(comparisons["online_failure"]["paired_episodes"], 1)
        self.assertEqual(
            comparisons["online_failure"]["success_rate_delta_a_minus_b"], 1.0
        )

    def test_duplicate_comparison_key_is_rejected(self) -> None:
        record = episode("a", "failure_start", True, "s0", 0)
        with self.assertRaisesRegex(ValueError, "duplicate paired comparison key"):
            score_records([record, dict(record)])


if __name__ == "__main__":
    unittest.main()
