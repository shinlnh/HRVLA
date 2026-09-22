"""Retraining must not reuse the incompatible historical 43-DoF dataset."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from hrvla_subtask.humanoidarena_retraining import GR00TRetrainJob, build_command


class TestGR00TRetrainContract(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.dataset = root / "dataset"
        (self.dataset / "meta").mkdir(parents=True)
        (self.dataset / "data").mkdir()
        self.info = self.dataset / "meta/info.json"
        self.info.write_text(json.dumps({"features": {
            "action": {"shape": [40]}, "observation.state": {"shape": [64]},
        }}), encoding="utf-8")
        (self.dataset / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"skill"}\n')
        self.base = root / "base"
        self.base.mkdir()
        (self.base / "config.json").touch()
        (self.base / "model.safetensors.index.json").touch()
        self.python = root / "python"
        self.script = root / "train.py"
        self.modality = root / "modality.py"
        for path in (self.python, self.script, self.modality):
            path.touch()
        self.job = GR00TRetrainJob(
            method_id="gr00t_st_rt", python=self.python, train_script=self.script,
            dataset=self.dataset, base_checkpoint=self.base,
            modality_config=self.modality, output_dir=root / "output", seed=0,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_builds_locked_gr00t_adapter_command(self) -> None:
        command = build_command(self.job)
        self.assertIn("--base-model-path", command)
        self.assertIn(str(self.dataset), command)
        self.assertIn("--max-steps", command)

    def test_rejects_legacy_43d_actions(self) -> None:
        self.info.write_text(json.dumps({"features": {
            "action": {"shape": [43]}, "observation.state": {"shape": [64]},
        }}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "action is not"):
            build_command(self.job)

    def test_str_requires_recovery_labels(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "recovery-label"):
            build_command(replace(self.job, method_id="gr00t_str_rt"))
