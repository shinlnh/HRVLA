"""Fail-closed single- and batched-episode PI0.5 STR nominal contracts."""

from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile

from hrvla_bench.humanoidarena_method_runtime import load_method_programs
from scripts.run_humanoidarena_pi05_str_episode import TASK_IDS, validate_invocation


ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = load_method_programs(ROOT / "benchmark/humanoidarena_method_programs.json")


class TestPI05STRNominalRunner(unittest.TestCase):
    def arguments(self, task: str) -> list[str]:
        return [
            "--task", task, "--seed", "7",
            "--lerobot_server_url", "http://127.0.0.1:8765",
            "--enable_cameras",
        ]

    def test_all_seven_task_routes(self) -> None:
        self.assertEqual(len(TASK_IDS), 7)
        for gym_id, program_id in TASK_IDS.items():
            with self.subTest(program_id=program_id):
                self.assertEqual(validate_invocation(self.arguments(gym_id), PROGRAMS), program_id)

    def test_refuses_malformed_batch_and_non_loopback_policy(self) -> None:
        args = self.arguments(next(iter(TASK_IDS)))
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory) / "jobs.json"
            batch.write_text(json.dumps({"episodes": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain episodes"):
                validate_invocation(args + ["--episode_batch_json", str(batch)], PROGRAMS)
        with self.assertRaisesRegex(ValueError, "loopback"):
            validate_invocation(
                ["http://remote:8765" if value == "http://127.0.0.1:8765" else value
                 for value in args], PROGRAMS,
            )

    def test_accepts_unique_batch_and_rejects_duplicate_index(self) -> None:
        args = self.arguments(next(iter(TASK_IDS)))
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory) / "jobs.json"
            episodes = [
                {"episode_index": 0, "episode_seed": 11, "result_json": "one.json"},
                {"episode_index": 1, "episode_seed": 12, "result_json": "two.json"},
            ]
            batch.write_text(json.dumps({"episodes": episodes}), encoding="utf-8")
            self.assertEqual(
                validate_invocation(args + ["--episode_batch_json", str(batch)], PROGRAMS),
                TASK_IDS[args[1]],
            )
            episodes[1]["episode_index"] = 0
            batch.write_text(json.dumps({"episodes": episodes}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "indices must be unique"):
                validate_invocation(args + ["--episode_batch_json", str(batch)], PROGRAMS)

    def test_refuses_missing_camera_and_unknown_task(self) -> None:
        args = self.arguments(next(iter(TASK_IDS)))
        with self.assertRaisesRegex(ValueError, "front camera"):
            validate_invocation(args[:-1], PROGRAMS)
        with self.assertRaisesRegex(ValueError, "outside the seven-task"):
            validate_invocation(["unknown" if value == args[1] else value for value in args], PROGRAMS)

if __name__ == "__main__":
    unittest.main()
