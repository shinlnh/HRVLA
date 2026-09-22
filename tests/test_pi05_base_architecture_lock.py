"""PI0.5 base architecture must retain its exact live HA interface and locks."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from scripts.run_humanoidarena_baseline_matrix_fast import MODES, TASKS


ROOT = Path(__file__).resolve().parents[1]
LOCKS = ROOT / "benchmark/locks/pi05_ha"


class PI05BaseArchitectureLockTest(unittest.TestCase):
    def test_seven_task_checkpoint_and_sonic_contract(self) -> None:
        checkpoint = json.loads((LOCKS / "checkpoints.json").read_text())
        sonic = json.loads((LOCKS / "sonic.json").read_text())
        expected = set(TASKS) - {"open_door"} | {"open_door_diagnostic"}
        self.assertEqual(set(checkpoint["weights_sha256_by_task"]), expected)
        self.assertEqual(set(checkpoint["small_file_manifest_sha256_by_task"]), expected)
        self.assertEqual(checkpoint["weight_size_bytes"], 9354116320)
        self.assertEqual(len(sonic["encoder_sha256"]), 64)
        self.assertEqual(len(sonic["decoder_sha256"]), 64)

    def test_six_task_primary_excludes_diagnostic_open_door(self) -> None:
        protocol = json.loads((LOCKS / "protocol.json").read_text())
        seeds = json.loads((LOCKS / "seed_plan.json").read_text())
        self.assertEqual(set(protocol["primary_tasks"]), set(TASKS) - {"open_door"})
        self.assertEqual(protocol["diagnostic_tasks"], ["open_door"])
        self.assertEqual(protocol["modes"], list(MODES))
        self.assertEqual(protocol["observation_dim"], 64)
        self.assertEqual(protocol["action_dim"], 40)
        self.assertEqual(seeds["identity_count"], 1440)


if __name__ == "__main__":
    unittest.main()
