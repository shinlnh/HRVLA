"""Fail-closed command contracts for the live PI0.5-STR runner."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import types
import unittest

from scripts.run_humanoidarena_pi05_str_smoke import (
    _scenario,
    _start_snapshot,
    build_sim_command,
)


TASK_GYM = "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby"


class TestPI05STRLiveRunner(unittest.TestCase):
    def test_command_uses_live_recovery_wrapper_and_action40_policy(self) -> None:
        evaluator = Path("/pinned/evaluator.py")

        def sim_command(**kwargs):
            return [
                "/sim/python", "-u", str(evaluator),
                "--task", TASK_GYM, "--seed", "0", "--max_steps", "4500",
                "--lerobot_server_url", "http://127.0.0.1:8765",
                "--episode_batch_json", str(kwargs["batch_path"]), "--enable_cameras",
            ]

        baseline = types.SimpleNamespace(
            SIM_EVALUATOR=evaluator,
            _sim_command=sim_command,
            _derive_episode_seed=lambda *_args: 73,
            TASKS={"pp_box": {"task_id": TASK_GYM, "model": "pi/model"}},
        )
        scenario = {"id": "box-missed-grasp-retry", "protocol": "online_failure"}
        command = build_sim_command(
            baseline, runtime_root=Path("/runtime"), suite_path=Path("/suite.json"),
            scenario=scenario, task_key="pp_box", output=Path("/output"),
            snapshot_path=Path("/capture/initial.json"), snapshot_sha256="abc",
            port=8765, seed=0, max_steps=120,
        )
        self.assertNotIn("--episode_batch_json", command)
        self.assertEqual(command[command.index("--internal-method") + 1], "pi05_str")
        self.assertEqual(command[command.index("--recovery-scenario") + 1], scenario["id"])
        self.assertEqual(command[command.index("--runtime-root") + 1], "/runtime")
        self.assertEqual(command[command.index("--episode_seed") + 1], "73")
        self.assertEqual(command[command.index("--max_steps") + 1], "120")
        self.assertIn("--enable_cameras", command)

    def test_snapshot_selection_is_protocol_locked(self) -> None:
        suite = {
            "tasks": [{
                "id": "pick_and_place_box", "humanoidarena_task_key": "pp_box",
                "scenarios": [
                    {"id": "online", "protocol": "online_failure"},
                    {"id": "failure", "protocol": "failure_start"},
                ],
            }]
        }
        task, online = _scenario(suite, "online")
        self.assertEqual(task["id"], "pick_and_place_box")
        _, failure = _scenario(suite, "failure")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "initial.json").write_text("{}", encoding="utf-8")
            (root / "failure.json").write_text("{}", encoding="utf-8")
            manifest = {"scenarios": [
                {"scenario_id": "online", "initial_snapshot_path": "initial.json",
                 "initial_snapshot_sha256": "initial-hash"},
                {"scenario_id": "failure", "failure_snapshot_path": "failure.json",
                 "failure_snapshot_sha256": "failure-hash"},
            ]}
            self.assertEqual(_start_snapshot(manifest, online, root)[1], "initial-hash")
            self.assertEqual(_start_snapshot(manifest, failure, root)[1], "failure-hash")

    def test_rejects_missing_capture_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "has no missing row"):
                _start_snapshot(
                    {"scenarios": []},
                    {"id": "missing", "protocol": "online_failure"},
                    Path(directory),
                )


if __name__ == "__main__":
    unittest.main()
