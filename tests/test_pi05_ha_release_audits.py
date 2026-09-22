"""Regression gates for the completed PI0.5 HumanoidArena baseline branch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import audit_pi05_ha_visuals as visuals
from scripts.audit_pi05_ha_baseline_release import canonical_sha256, require_hash
from scripts.benchmark_v2_preflight import inspect_contract


ROOT = Path(__file__).resolve().parents[1]


class PI05HABaselineReleaseTest(unittest.TestCase):
    def test_complete_contract_passes_fail_closed_preflight(self) -> None:
        contract = json.loads((ROOT / "benchmark/v2_contract.json").read_text())
        self.assertEqual(contract["status"], "complete")
        self.assertEqual(
            inspect_contract(
                contract, ROOT,
                branch="feat(PI05-benchmark-HA)/evaluate",
                architecture_head=contract["architecture_revision"],
                architecture_in_history=True,
                suite_head=contract["suite_revision"],
            ),
            [],
        )

    def test_locked_hash_rejects_changed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            path.write_text("original", encoding="utf-8")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            require_hash(path, expected)
            path.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash-mismatched"):
                require_hash(path, expected)

    def test_seed_identity_digest_is_order_sensitive(self) -> None:
        one = [{"task": "boxing", "seed": 0}, {"task": "boxing", "seed": 1}]
        self.assertNotEqual(canonical_sha256(one), canonical_sha256(list(reversed(one))))

    def test_visual_copy_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_dir = root / "videos"
            video_dir.mkdir()
            selected = []
            for task in visuals.TASKS:
                payload = (task * 200).encode()
                copy = video_dir / f"{task}.mp4"
                copy.write_bytes(payload)
                digest = hashlib.sha256(payload).hexdigest()
                for index in range(24):
                    selected.append({
                        "path": f"raw/{task}/sample-{index:02d}.mp4",
                        "sha256": digest,
                    })
            summary = root / "summary.json"
            summary.write_text(json.dumps({"selected_video_manifest": selected}))
            with mock.patch.object(visuals, "SUMMARY", summary), mock.patch.object(visuals, "VIDEOS", video_dir), mock.patch.object(visuals, "ROOT", root):
                self.assertEqual(visuals.audit()["checked_in_video_count"], 7)
                (video_dir / "boxing.mp4").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "differs from frozen source"):
                    visuals.audit()


if __name__ == "__main__":
    unittest.main()
