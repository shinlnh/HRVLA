from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

try:
    import numpy as np

    from prepare_vla_retraining_data import phase_labels, split_episode_ids  # noqa: E402
except ModuleNotFoundError:
    np = None
    phase_labels = None
    split_episode_ids = None


class VLARetrainingLockTest(unittest.TestCase):
    def test_lock_is_consistent(self) -> None:
        lock = json.loads(
            (REPO_ROOT / "config/subtask_vla_retraining.lock.json").read_text(
                encoding="utf-8"
            )
        )
        dataset = lock["dataset"]
        training = lock["training"]
        self.assertEqual(
            dataset["source_episodes"],
            dataset["train_episodes"] + dataset["heldout_episodes"],
        )
        self.assertEqual(
            training["effective_batch_size"],
            training["physical_batch_size"]
            * training["gradient_accumulation_steps"],
        )
        self.assertEqual(lock["evaluation"]["trajectory_count"], 42)

    def test_recovery_lock_is_consistent_when_present(self) -> None:
        path = REPO_ROOT / "config/recovery_vla_retraining.lock.json"
        if not path.exists():
            self.skipTest("recovery retraining profile is not present on this branch")
        lock = json.loads(path.read_text(encoding="utf-8"))
        dataset = lock["dataset"]
        training = lock["training"]
        self.assertEqual(dataset["label_mode"], "counterfactual_recovery_conditioned")
        self.assertEqual(
            dataset["source_episodes"],
            dataset["train_episodes"] + dataset["heldout_episodes"],
        )
        self.assertEqual(
            training["effective_batch_size"],
            training["physical_batch_size"]
            * training["gradient_accumulation_steps"],
        )


@unittest.skipUnless(np is not None, "VLA data preparation extras are not installed")
class VLADataPreparationTest(unittest.TestCase):
    def test_split_is_disjoint_deterministic_and_complete(self) -> None:
        first = split_episode_ids(range(20), 0.2, 42)
        second = split_episode_ids(range(20), 0.2, 42)
        self.assertEqual(first, second)
        train, heldout = first
        self.assertEqual(len(heldout), 4)
        self.assertFalse(set(train) & set(heldout))
        self.assertEqual(set(train) | set(heldout), set(range(20)))

    def test_phase_labels_follow_prediction_horizon(self) -> None:
        actions = np.zeros((120, 43), dtype=np.float32)
        actions[50:100, 22:29] = 1.0
        labels, boundary = phase_labels(actions, horizon=10)
        self.assertEqual(boundary["close_at"], 50)
        self.assertEqual(boundary["release_at"], 100)
        self.assertEqual(labels[39], 0)
        self.assertEqual(labels[41], 1)
        self.assertEqual(labels[50], 2)
        self.assertEqual(labels[91], 3)

    def test_invalid_fraction_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            split_episode_ids(range(3), 1.0, 0)


if __name__ == "__main__":
    unittest.main()
