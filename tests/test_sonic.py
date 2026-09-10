import json
from pathlib import Path
import tempfile
import unittest

from hrvla_bench.sonic import disturbance_pilot_report


def metrics(local_mpjpe: float) -> dict:
    return {
        "eval/all_metrics_dict": {
            "motion_keys": ["motion-a"],
            "terminated": [False],
            "progress": [1.0],
            "mpjpe_g": [local_mpjpe + 2.0],
            "mpjpe_l": [local_mpjpe],
            "mpjpe_pa": [local_mpjpe - 2.0],
        }
    }


class SonicPilotTests(unittest.TestCase):
    def test_report_requires_and_preserves_injection_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nominal = root / "nominal.json"
            disturbed = root / "disturbed.json"
            audit = root / "audit.jsonl"
            nominal.write_text(json.dumps(metrics(10.0)), encoding="utf-8")
            disturbed.write_text(json.dumps(metrics(12.0)), encoding="utf-8")
            audit.write_text(
                json.dumps(
                    {
                        "event": "push_once_by_setting_velocity",
                        "environment_ids": [0],
                        "episode_steps": [100],
                        "velocity_range": {"y": [0.35, 0.35]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = disturbance_pilot_report(
                nominal,
                disturbed,
                audit,
                checkpoint_id="checkpoint",
                controller_id="controller",
                simulator_revision="simulator",
                run_id="pilot",
                recorded_at="2026-09-10T07:04:26+00:00",
            )
        self.assertTrue(report["injection_verified"])
        self.assertEqual(report["recorded_at"], "2026-09-10T07:04:26+00:00")
        self.assertEqual(report["artifacts"]["nominal_metrics"]["path"], str(nominal))
        self.assertEqual(report["claim_status"], "diagnostic_only_unadmitted")
        self.assertEqual(
            report["delta_disturbed_minus_nominal"]["mean_mpjpe_local_mm"], 2.0
        )


if __name__ == "__main__":
    unittest.main()
