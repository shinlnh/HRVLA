import copy
import unittest
from pathlib import Path

from hrvla_bench.evidence import audit_evidence, claim_readiness
from hrvla_bench.plan import build_plan, load_json


SUITE = Path(__file__).resolve().parents[1] / "benchmark/suites/hrvla_recovery_v0.json"


def admitted_suite() -> dict:
    suite = {
        "schema_version": "1.0",
        "suite_id": "proof-test",
        "controller_contract": {},
        "replication": {"training_seeds": [0], "rollouts_per_seed_per_cell": 2},
        "tasks": [
            {
                "id": "pick",
                "initial_snapshot_id": "snapshot:initial",
                "success_predicate": "object placed",
                "horizon_s": 30,
                "admission": {
                    "status": "admitted",
                    "initial_snapshot_sha256": "a" * 64,
                    "predicate_test_id": "tests/pick-success@abc123",
                },
                "scenarios": [
                    {
                        "id": "drop",
                        "protocol": "failure_start",
                        "failure_snapshot_id": "snapshot:drop",
                        "event_id": "after-grasp",
                        "event_boundary": "stable grasp",
                        "level": "L3",
                        "humanoid_axes": ["H3"],
                        "severity": "medium",
                        "injector": {"id": "drop-object", "parameters": {}},
                        "admission": {
                            "status": "admitted",
                            "oracle_id": "oracle@abc123",
                            "success_rate": 1.0,
                            "oracle_trials": 20,
                            "oracle_successes": 20,
                            "injector_revision": "abc123",
                            "predicate_test_id": "tests/drop-success@abc123",
                            "failure_snapshot_sha256": "b" * 64,
                        },
                    }
                ],
            }
        ],
    }
    return suite


def records_for(plan: dict) -> list[dict]:
    records = []
    for method in plan["methods"]:
        for index, episode in enumerate(plan["episodes"]):
            record = {
                "schema_version": "1.0",
                "run_id": f"{method}-run",
                "episode_id": f"{method}-{index}",
                "method_id": method,
                "evaluation_track": "end_to_end_recovery",
                "suite_id": plan["suite_id"],
                "suite_sha256": plan["suite_sha256"],
                "plan_sha256": plan["plan_sha256"],
                "task_id": episode["task_id"],
                "scenario_id": episode["scenario_id"],
                "protocol": episode["protocol"],
                "training_seed": episode["training_seed"],
                "rollout_seed": episode["rollout_seed"],
                "initial_snapshot_id": episode["initial_snapshot_id"],
                "failure_snapshot_id": episode.get("failure_snapshot_id"),
                "failure": episode.get("failure"),
                "policy_checkpoint_id": f"{method}-seed-{episode['training_seed']}",
                "controller_id": "sonic@locked",
                "simulator_revision": "isaac-sim@locked",
                "success": True,
                "termination": "success",
            }
            records.append(record)
    return records


class ReadinessTests(unittest.TestCase):
    def test_repository_suite_reports_draft_blockers(self) -> None:
        report = claim_readiness(load_json(SUITE))
        self.assertFalse(report["claim_ready"])
        self.assertEqual(report["admitted_tasks"], 0)
        self.assertIn("suite has no admitted recovery scenarios", report["blockers"])

    def test_fully_admitted_suite_is_ready(self) -> None:
        report = claim_readiness(admitted_suite())
        self.assertTrue(report["claim_ready"])
        self.assertEqual(report["admitted_tasks"], 1)
        self.assertEqual(report["admitted_scenarios"], 1)


class EvidenceAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = build_plan(admitted_suite(), ["baseline", "candidate"])
        self.records = records_for(self.plan)

    def test_complete_paired_evidence_is_claim_eligible(self) -> None:
        report = audit_evidence(self.plan, self.records)
        self.assertTrue(report["claim_eligible"])
        self.assertEqual(report["episodes_per_method"], 4)
        self.assertEqual(report["episode_records"], 8)

    def test_missing_method_record_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete paired evidence"):
            audit_evidence(self.plan, self.records[:-1])

    def test_controller_change_is_rejected(self) -> None:
        records = copy.deepcopy(self.records)
        records[-1]["controller_id"] = "sonic@changed"
        with self.assertRaisesRegex(ValueError, "controller or simulator differs"):
            audit_evidence(self.plan, records)

    def test_failure_contract_change_is_rejected(self) -> None:
        records = copy.deepcopy(self.records)
        recovery = next(record for record in records if record["failure"] is not None)
        recovery["failure"]["injector_parameters"] = {"hidden_change": True}
        with self.assertRaisesRegex(ValueError, "failure contract does not match"):
            audit_evidence(self.plan, records)

    def test_checkpoint_change_within_seed_is_rejected(self) -> None:
        records = copy.deepcopy(self.records)
        records[1]["policy_checkpoint_id"] = "baseline-seed-0-changed"
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            audit_evidence(self.plan, records)


if __name__ == "__main__":
    unittest.main()
