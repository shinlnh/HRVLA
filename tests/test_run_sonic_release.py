from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_sonic_release.py"
SPEC = importlib.util.spec_from_file_location("run_sonic_release", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class RunSonicReleaseTest(unittest.TestCase):
    def test_hydra_vector_is_shell_independent(self) -> None:
        self.assertEqual(runner.hydra_vector((4.5, 0.0, -1.25)), "[4.5,0,-1.25]")


if __name__ == "__main__":
    unittest.main()
