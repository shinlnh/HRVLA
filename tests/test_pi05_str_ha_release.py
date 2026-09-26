"""Release integrity tests for benchmark #3."""

from pathlib import Path
import tempfile
import unittest

from scripts.audit_pi05_str_ha_release import PACKAGE, audit


class TestPI05STRHARelease(unittest.TestCase):
    def test_checked_in_release_passes(self) -> None:
        result = audit(PACKAGE)
        self.assertTrue(result["audit_passed"])
        self.assertEqual(result["episodes_observed"], 1440)
        self.assertEqual(result["checked_in_videos"], 6)

    def test_missing_manifest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                audit(Path(directory))


if __name__ == "__main__":
    unittest.main()
