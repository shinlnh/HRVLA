from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hrvla_subtask.model import ExecutionMemory, TaskSpec  # noqa: E402
from hrvla_subtask.simulator import load_suite  # noqa: E402


class TaskModelTest(unittest.TestCase):
    def test_suite_is_valid_and_reachable(self) -> None:
        tasks = load_suite(REPO_ROOT / "benchmark" / "subtask_suite_v1.json")
        self.assertEqual(len(tasks), 6)
        self.assertTrue(all(task.reachable() for task in tasks))

    def test_memory_catches_up_and_rolls_back(self) -> None:
        task = load_suite(REPO_ROOT / "benchmark" / "subtask_suite_v1.json")[0]
        memory = ExecutionMemory()
        first = task.canonical_next(task.initial_state)
        assert first is not None
        state = first.apply(task.initial_state)
        self.assertEqual(memory.reconcile(task, state), "catch_up")
        self.assertIn(first.skill_id, memory.completed)
        rolled_back = frozenset(item for item in state if item != first.milestone)
        self.assertEqual(memory.reconcile(task, rolled_back), "rollback")
        self.assertNotIn(first.skill_id, memory.completed)

    def test_lock_file_is_machine_readable(self) -> None:
        lock = json.loads((REPO_ROOT / "config" / "subtask-planner.lock.json").read_text())
        self.assertEqual(lock["proposal_model"]["revision"], "9ce19a195e423419c349abfc86fd07178b230561")


if __name__ == "__main__":
    unittest.main()
