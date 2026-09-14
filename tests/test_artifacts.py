import tempfile
import unittest
from pathlib import Path

from hrvla_bench.artifacts import build_artifact_lock, verify_artifact_lock


class ArtifactLockTests(unittest.TestCase):
    def test_lock_is_deterministic_and_verifiable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "artifacts" / "checkpoint-final"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
            (checkpoint / "weights.bin").write_bytes(b"weights")
            groups = {"method": "artifacts/checkpoint-final"}
            first = build_artifact_lock(root, groups)
            second = build_artifact_lock(root, groups)
            self.assertEqual(first, second)
            self.assertEqual(verify_artifact_lock(first, root), [])

    def test_tampered_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            artifact = checkpoint / "weights.bin"
            artifact.write_bytes(b"original")
            lock = build_artifact_lock(root, {"method": "checkpoint"})
            artifact.write_bytes(b"tampered")
            errors = verify_artifact_lock(lock, root)
            self.assertTrue(any("mismatch" in error for error in errors))

    def test_unsafe_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unsafe artifact root"):
                build_artifact_lock(Path(directory), {"method": "../outside"})


if __name__ == "__main__":
    unittest.main()
