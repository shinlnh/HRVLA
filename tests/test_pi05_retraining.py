"""The RT branch must not silently train on incompatible data or weights."""

import json
from pathlib import Path
import tempfile
import unittest

from hrvla_bench.pi05_retraining import PI05TrainingJob, build_training_command


class TestPI05Retraining(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.dataset = root / "dataset"
        (self.dataset / "meta").mkdir(parents=True)
        (self.dataset / "data").mkdir()
        (self.dataset / "meta" / "tasks.parquet").touch()
        self.info = self.dataset / "meta" / "info.json"
        self.info.write_text(json.dumps({
            "codebase_version": "v3.0",
            "features": {"action": {"shape": [40]}, "observation.state": {"shape": [64]}},
        }), encoding="utf-8")
        self.policy = root / "policy"
        self.policy.mkdir()
        (self.policy / "config.json").write_text('{"type":"pi05"}', encoding="utf-8")
        self.python = root / "python"
        self.python.touch()
        self.script = root / "lerobot_train.py"
        self.script.touch()
        self.job = PI05TrainingJob(
            method_id="pi05_st_rt", dataset=self.dataset, base_policy=self.policy,
            output_dir=root / "output", train_script=self.script, python=self.python,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_command_is_gpu_training_on_local_dataset(self) -> None:
        command = build_training_command(self.job)
        self.assertIn("--policy.device=cuda", command)
        self.assertIn("--policy.gradient_checkpointing=true", command)
        self.assertIn(f"--dataset.root={self.dataset}", command)
        self.assertIn("--policy.push_to_hub=false", command)

    def test_rejects_old_dataset_without_mutating_it(self) -> None:
        payload = json.loads(self.info.read_text(encoding="utf-8"))
        payload["codebase_version"] = "v2.1"
        self.info.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "convert a copy"):
            build_training_command(self.job)
        self.assertEqual(json.loads(self.info.read_text())["codebase_version"], "v2.1")

    def test_rejects_wrong_action_contract(self) -> None:
        payload = json.loads(self.info.read_text(encoding="utf-8"))
        payload["features"]["action"]["shape"] = [32]
        self.info.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "action must have shape"):
            build_training_command(self.job)

    def test_rejects_nonempty_output(self) -> None:
        self.job.output_dir.mkdir()
        (self.job.output_dir / "checkpoint").touch()
        with self.assertRaises(FileExistsError):
            build_training_command(self.job)
