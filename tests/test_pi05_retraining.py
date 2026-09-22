"""The RT branch must not silently train on incompatible data or weights."""

import json
import hashlib
from dataclasses import replace
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
        self.assertIn("--policy.train_expert_only=true", command)
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

    def test_verified_export_selects_only_task_matched_episodes(self) -> None:
        info = json.loads(self.info.read_text(encoding="utf-8"))
        info["hrvla_packed_video_export"] = {"schema_version": 1}
        info["hrvla_subtask_relabel"] = {"split": "train"}
        self.info.write_text(json.dumps(info), encoding="utf-8")
        provenance = self.dataset / "meta/hrvla_source_episodes.jsonl"
        provenance.write_text("".join(json.dumps(row) + "\n" for row in (
            {"episode_index": 0, "source_dataset": "HOI_pp_box/sonic_refpose_v3_1"},
            {"episode_index": 1, "source_dataset": "HSI_boxing/sonic_refpose_v3_1"},
            {"episode_index": 2, "source_dataset": "HOI_pp_box/sonic_refpose_v3_1"},
        )), encoding="utf-8")
        digest = hashlib.sha256(provenance.read_bytes()).hexdigest()
        (self.dataset / "meta/hrvla_export_audit.json").write_text(
            json.dumps({"status": "loader_pass", "provenance_sha256": digest}),
            encoding="utf-8",
        )
        policy = self.policy.parent / "HOI_pp_box" / "policy"
        policy.mkdir(parents=True)
        (policy / "config.json").write_text('{"type":"pi05"}', encoding="utf-8")
        job = replace(self.job, base_policy=policy, source_dataset="HOI_pp_box")
        self.assertIn("--dataset.episodes=[0,2]", build_training_command(job))
        with self.assertRaisesRegex(ValueError, "task-matched"):
            build_training_command(replace(job, source_dataset="HSI_boxing"))
        (self.dataset / "meta/hrvla_export_audit.json").write_text(
            json.dumps({"status": "structural_pass_loader_pending", "provenance_sha256": digest}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "loader audit"):
            build_training_command(job)
