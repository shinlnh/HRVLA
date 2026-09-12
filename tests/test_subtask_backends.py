from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hrvla_subtask.backends import CosmosReasonBackend  # noqa: E402
from hrvla_subtask.model import ExecutionMemory  # noqa: E402
from hrvla_subtask.simulator import load_suite  # noqa: E402


class CosmosBackendTest(unittest.TestCase):
    def test_extracts_plain_and_fenced_json(self) -> None:
        plain = CosmosReasonBackend._extract_json(
            'analysis {"skill_id":"grasp_apple","confidence":0.8}'
        )
        fenced = CosmosReasonBackend._extract_json(
            '```json\n{"skill_id":"place_apple","confidence":0.7}\n```'
        )
        self.assertEqual(plain["skill_id"], "grasp_apple")
        self.assertEqual(fenced["skill_id"], "place_apple")

    def test_rejects_malformed_json(self) -> None:
        self.assertIsNone(CosmosReasonBackend._extract_json("{not-json}"))

    def test_prompt_contains_current_executable_action_mask(self) -> None:
        backend = CosmosReasonBackend.__new__(CosmosReasonBackend)
        task = load_suite(REPO_ROOT / "benchmark" / "subtask_suite_v1.json")[0]
        prompt = backend._prompt(task, task.initial_state, ExecutionMemory(), 1)
        executable = prompt.split("EXECUTABLE_NOW: ", 1)[1].split("\n", 1)[0]
        self.assertIn('"id": "grasp_apple"', executable)
        self.assertIn('"id": "close_drawer"', executable)
        self.assertNotIn('"id": "place_apple"', executable)


if __name__ == "__main__":
    unittest.main()
