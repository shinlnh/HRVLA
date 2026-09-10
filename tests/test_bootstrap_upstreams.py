from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bootstrap = load_script("bootstrap_upstreams.py")


class BootstrapUpstreamsTest(unittest.TestCase):
    def test_checkout_materializes_tree_when_remote_head_matches_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            seed = root / "seed"
            origin = root / "origin.git"
            vendor = root / "vendor"

            subprocess.run(["git", "init", "-q", str(seed)], check=True)
            subprocess.run(
                ["git", "config", "user.email", "tests@hrvla.local"],
                cwd=seed,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "HRVLA tests"],
                cwd=seed,
                check=True,
            )
            (seed / "tracked.txt").write_text("locked tree\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=seed, check=True)
            subprocess.run(["git", "commit", "-qm", "seed"], cwd=seed, check=True)
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=seed,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            subprocess.run(["git", "clone", "-q", "--bare", str(seed), str(origin)], check=True)

            bootstrap.checkout_component(
                "fixture",
                {
                    "url": str(origin),
                    "revision": revision,
                    "destination": "fixture",
                },
                vendor,
                pull_lfs=False,
            )

            checkout = vendor / "fixture"
            self.assertEqual((checkout / "tracked.txt").read_text(), "locked tree\n")
            self.assertEqual(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=checkout,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout,
                "",
            )


if __name__ == "__main__":
    unittest.main()
