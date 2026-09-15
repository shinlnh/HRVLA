from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from hrvla_bench.humanoidarena_release import build_release_manifest, file_sha256


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_humanoidarena_internal_matrix.py"
SPEC = importlib.util.spec_from_file_location("internal_matrix_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_internal_command_composes_method_and_recovery_hooks(tmp_path: Path) -> None:
    command = RUNNER.build_internal_command(
        method_id="gr00t_str",
        task_key="football",
        scenario_id="support-state-push",
        suite_path=tmp_path / "suite.json",
        method_programs_path=tmp_path / "programs.json",
        attempt_dir=tmp_path / "attempt-0001",
        snapshot_path=tmp_path / "initial.json",
        snapshot_sha256="a" * 64,
        batch_path=tmp_path / "batch.json",
        port=18445,
        record_video_every_n=13,
    )
    assert str(RUNNER.INTERNAL_EPISODE_RUNNER) in command
    assert command[command.index("--internal-method") + 1] == "gr00t_str"
    assert command[command.index("--recovery-scenario") + 1] == "support-state-push"
    assert command[command.index("--record_video_every_n") + 1] == "13"
    assert "--per-episode-output" in command


def test_nominal_internal_command_does_not_select_recovery_scenario(tmp_path: Path) -> None:
    command = RUNNER.build_internal_command(
        method_id="gr00t_sonic",
        task_key="boxing",
        scenario_id=None,
        suite_path=tmp_path / "suite.json",
        method_programs_path=tmp_path / "programs.json",
        attempt_dir=tmp_path / "attempt-0001",
        snapshot_path=tmp_path / "initial.json",
        snapshot_sha256="b" * 64,
        batch_path=tmp_path / "batch.json",
        port=18445,
        record_video_every_n=13,
    )
    assert "--recovery-scenario" not in command


def test_checkpoint_server_uses_the_frozen_device_and_denoising_steps() -> None:
    command = RUNNER._server_command(
        {"path": "checkpoints/common-seed-0"}, port=18445, device="cpu"
    )
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--denoising-steps") + 1] == "4"


def test_checkpoint_verifier_hashes_every_manifest_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime.py"
    runtime.write_text("# frozen\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoints/common-seed-0"
    checkpoint.mkdir(parents=True)
    weights = checkpoint / "weights.bin"
    weights.write_bytes(b"weights")
    manifest = build_release_manifest(
        tmp_path,
        artifact_id="common-seed-0",
        artifact_type="checkpoint",
        root="checkpoints/common-seed-0",
        repo_id="owner/repo",
        repo_type="model",
        path_in_repo="humanoidarena-v1/common-seed-0",
        source_revision="a" * 40,
        provenance={"training_seed": 0},
    )
    manifest_path = tmp_path / "manifests/common-seed-0.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    row = {
        "checkpoint_id": "immutable",
        "path": "checkpoints/common-seed-0",
        "manifest_path": "manifests/common-seed-0.json",
        "manifest_sha256": manifest["manifest_sha256"],
    }
    lock = {
        "method_runtime_signatures": {"runtime.py": file_sha256(runtime)},
        "checkpoints": {
            method: {"0": row} for method in RUNNER.INTERNAL_METHODS
        },
    }
    monkeypatch.setattr(RUNNER, "ROOT", tmp_path)
    RUNNER.verify_checkpoint_files(lock)
    weights.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint artifact differs"):
        RUNNER.verify_checkpoint_files(lock)
