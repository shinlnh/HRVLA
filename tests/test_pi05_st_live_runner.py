"""Fail-closed single-episode contract for the PI0.5 ST Isaac hook."""

from __future__ import annotations

import unittest
from pathlib import Path
import types

from hrvla_bench.humanoidarena_method_runtime import load_method_programs
from scripts.run_humanoidarena_pi05_st_episode import TASK_IDS, validate_invocation
from scripts.run_humanoidarena_pi05_st_smoke import build_sim_command


ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = load_method_programs(ROOT / "benchmark/humanoidarena_method_programs.json")


class TestPI05STLiveRunner(unittest.TestCase):
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

    def test_refuses_batch_and_non_loopback_policy(self) -> None:
        args = self.arguments(next(iter(TASK_IDS)))
        with self.assertRaisesRegex(ValueError, "one episode"):
            validate_invocation(args + ["--episode_batch_json", "jobs.json"], PROGRAMS)
        with self.assertRaisesRegex(ValueError, "loopback"):
            validate_invocation(
                ["http://remote:8765" if value == "http://127.0.0.1:8765" else value
                 for value in args], PROGRAMS,
            )

    def test_refuses_missing_camera_and_unknown_task(self) -> None:
        args = self.arguments(next(iter(TASK_IDS)))
        with self.assertRaisesRegex(ValueError, "front camera"):
            validate_invocation(args[:-1], PROGRAMS)
        with self.assertRaisesRegex(ValueError, "outside the seven-task"):
            validate_invocation(["unknown" if value == args[1] else value for value in args], PROGRAMS)

    def test_smoke_command_keeps_pinned_evaluator_args_without_batch(self) -> None:
        def sim_command(**kwargs):
            return [
                "/sim/python", "-u", "/pinned/evaluator.py",
                "--task", TASK_GYM, "--seed", "0", "--max_steps", "1450",
                "--lerobot_server_url", "http://127.0.0.1:8765",
                "--episode_batch_json", str(kwargs["batch_path"]), "--enable_cameras",
            ]

        baseline = types.SimpleNamespace(
            _sim_command=sim_command,
            _derive_episode_seed=lambda *_args: 42,
            TASKS={"pp_box": {"task_id": TASK_GYM, "model": "pi/model"}},
        )
        command = build_sim_command(
            baseline, runtime_root=Path("/runtime"), task_key="pp_box",
            mode="base_test", output=Path("/output"), port=8765,
            seed=0, max_steps=120,
        )
        self.assertNotIn("--episode_batch_json", command)
        self.assertEqual(command[command.index("--max_steps") + 1], "120")
        self.assertEqual(command[command.index("--episode_seed") + 1], "42")
        self.assertIn("--method-output-dir", command)
        self.assertIn("--enable_cameras", command)


TASK_GYM = "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby"


if __name__ == "__main__":
    unittest.main()
