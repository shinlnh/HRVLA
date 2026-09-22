"""Readiness cannot be inferred from a branch name or a stale result file."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.benchmark_v2_preflight import inspect_contract


REVISION = "a" * 40
SUITE_REVISION = "b" * 40


class BenchmarkV2PreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        locks = {}
        for name in ("checkpoint_lock", "sonic_lock", "protocol_lock", "seed_plan_lock"):
            path = self.root / f"{name}.json"
            path.write_text('{"locked":true}', encoding="utf-8")
            locks[name] = {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        self.contract = {
            "schema_version": 1,
            "architecture_token": "GR00T",
            "architecture_branch": "feat(GR00T)/reproduce-gr00t-sonic-whole-body",
            "architecture_revision": REVISION,
            "suite": "HA",
            "suite_revision": SUITE_REVISION,
            "precision": "bf16",
            "expected_episodes": 2,
            "status": "ready",
            "result_manifest": None,
            **locks,
        }

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def inspect(self, **kwargs):
        return inspect_contract(
            self.contract, self.root,
            branch=kwargs.get("branch", "feat(GR00T-benchmark-HA)/evaluate"),
            architecture_head=kwargs.get("architecture_head", REVISION),
            architecture_in_history=kwargs.get("architecture_in_history", True),
            suite_head=kwargs.get("suite_head", SUITE_REVISION),
        )

    def test_exact_ready_contract_passes(self):
        self.assertEqual(self.inspect(), [])

    def test_pending_and_wrong_branch_fail(self):
        self.contract["status"] = "pending"
        errors = self.inspect(branch="feat(PI05-benchmark-HA)/evaluate")
        self.assertIn("branch remains pending", errors)
        self.assertIn("checked-out branch does not match the contract", errors)

    def test_changed_lock_and_stale_architecture_fail(self):
        (self.root / "checkpoint_lock.json").write_text("changed", encoding="utf-8")
        errors = self.inspect(architecture_head="c" * 40)
        self.assertIn("checkpoint_lock: SHA-256 mismatch", errors)
        self.assertIn("architecture revision is stale or absent from branch history", errors)

    def test_complete_requires_audited_exact_episode_set(self):
        self.contract["status"] = "complete"
        self.contract["result_manifest"] = "results.json"
        self.assertIn("complete branch lacks a valid result manifest", self.inspect())
        (self.root / "results.json").write_text(json.dumps({
            "episodes_observed": 1, "audit_passed": True,
        }), encoding="utf-8")
        self.assertIn("result manifest does not match expected episode count", self.inspect())
