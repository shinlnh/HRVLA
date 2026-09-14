import tempfile
import unittest
from pathlib import Path

from hrvla_bench.execution import RunIdentity, run_plan
from hrvla_bench.plan import build_plan, load_json


class NeverCalledAdapter:
    def run_episode(self, episode: dict) -> dict:
        raise AssertionError("draft refusal must happen before adapter execution")


class SuccessfulDiagnosticAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def run_episode(self, episode: dict) -> dict:
        self.calls += 1
        return {
            "success": True,
            "termination": "success",
            "completed_subtasks": 1,
            "total_subtasks": 1,
        }


class FailAfterFirstAdapter(SuccessfulDiagnosticAdapter):
    def run_episode(self, episode: dict) -> dict:
        if self.calls == 1:
            raise RuntimeError("simulated worker interruption")
        return super().run_episode(episode)


class ExecutionTests(unittest.TestCase):
    def _draft_plan(self) -> dict:
        suite_path = (
            Path(__file__).resolve().parents[1]
            / "benchmark/suites/hrvla_recovery_v0.json"
        )
        return build_plan(
            load_json(suite_path), ["method"], include_draft=True, rollouts_override=1
        )

    def test_draft_plan_is_refused(self) -> None:
        plan = self._draft_plan()
        identity = RunIdentity(
            "run", "method", "end_to_end_recovery", "checkpoint", "controller", "sim"
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "refusing non-admitted"):
                run_plan(plan, identity, NeverCalledAdapter(), Path(directory) / "out.jsonl")

    def test_explicit_diagnostic_draft_run_is_written(self) -> None:
        plan = self._draft_plan()
        identity = RunIdentity(
            "run", "method", "end_to_end_recovery", "checkpoint", "controller", "sim"
        )
        with tempfile.TemporaryDirectory() as directory:
            records = run_plan(
                plan,
                identity,
                SuccessfulDiagnosticAdapter(),
                Path(directory) / "out.jsonl",
                allow_draft=True,
            )
        self.assertEqual(len(records), 48)
        recovery = next(record for record in records if record["protocol"] != "nominal")
        self.assertFalse(recovery["failure"]["recoverable_oracle"])

    def test_interrupted_run_resumes_from_validated_partial_output(self) -> None:
        plan = self._draft_plan()
        identity = RunIdentity(
            "run", "method", "end_to_end_recovery", "checkpoint", "controller", "sim"
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.jsonl"
            with self.assertRaisesRegex(RuntimeError, "worker interruption"):
                run_plan(
                    plan,
                    identity,
                    FailAfterFirstAdapter(),
                    output,
                    allow_draft=True,
                )
            partial = output.with_suffix(".jsonl.partial")
            self.assertTrue(partial.is_file())
            self.assertEqual(len(partial.read_text().splitlines()), 1)

            resumed = SuccessfulDiagnosticAdapter()
            records = run_plan(
                plan, identity, resumed, output, allow_draft=True
            )
            self.assertEqual(resumed.calls, len(plan["episodes"]) - 1)
            self.assertEqual(len(records), len(plan["episodes"]))
            self.assertTrue(output.is_file())
            self.assertFalse(partial.exists())


if __name__ == "__main__":
    unittest.main()
