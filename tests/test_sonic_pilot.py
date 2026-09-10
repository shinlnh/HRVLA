from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_sonic_h1_pilot.py"
SPEC = importlib.util.spec_from_file_location("run_sonic_h1_pilot", SCRIPT)
assert SPEC and SPEC.loader
pilot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pilot
SPEC.loader.exec_module(pilot)


class SonicPilotCommandTest(unittest.TestCase):
    def test_overrides_keep_push_one_shot_and_audited(self) -> None:
        overrides = pilot.hydra_overrides(0.35, Path("/tmp/audit.jsonl"))
        self.assertIn("++manager_env.config.train_only_events=[]", overrides)
        self.assertIn(
            "++manager_env.events.push_robot.interval_range_s=[2.0,2.0]",
            overrides,
        )
        self.assertIn(
            "++manager_env.events.push_robot.params.velocity_range.y=[0.35,0.35]",
            overrides,
        )
        self.assertTrue(overrides[-1].endswith("/tmp/audit.jsonl"))


if __name__ == "__main__":
    unittest.main()
