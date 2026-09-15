import json
from pathlib import Path

import pytest

from hrvla_bench.humanoidarena_release import (
    RUNTIME_SIGNATURE_PATHS,
    build_release_manifest,
    release_receipt,
    validate_release_receipt,
    verify_release_manifest,
    write_json_once,
)


REVISION = "1" * 40


def test_runtime_signature_closure_binds_server_simulator_and_planner() -> None:
    paths = set(RUNTIME_SIGNATURE_PATHS)
    assert "scripts/serve_humanoidarena_gr00t.py" in paths
    assert "scripts/run_humanoidarena_internal_episode.py" in paths
    assert "src/hrvla_bench/isaac_events.py" in paths
    assert "src/hrvla_subtask/planner.py" in paths


def _manifest(root: Path) -> dict:
    artifact = root / "artifacts/checkpoint"
    artifact.mkdir(parents=True)
    (artifact / "config.json").write_text("{}\n", encoding="utf-8")
    (artifact / "weights.bin").write_bytes(b"weights")
    return build_release_manifest(
        root,
        artifact_id="common-seed-0",
        artifact_type="checkpoint",
        root="artifacts/checkpoint",
        repo_id="owner/repo",
        repo_type="model",
        path_in_repo="humanoidarena-v1/common/seed-0",
        source_revision=REVISION,
        provenance={"training_seed": 0, "selected_step": 200},
    )


def test_release_manifest_detects_tamper_and_unlocked_file(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert verify_release_manifest(manifest, tmp_path) == []
    (tmp_path / "artifacts/checkpoint/weights.bin").write_bytes(b"changed")
    errors = verify_release_manifest(manifest, tmp_path)
    assert any("size differs" in error or "SHA-256 differs" in error for error in errors)
    (tmp_path / "artifacts/checkpoint/extra.bin").write_bytes(b"extra")
    assert any("unlocked release file" in error for error in verify_release_manifest(manifest, tmp_path))


def test_release_receipt_binds_manifest_and_immutable_revision(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    receipt = release_receipt(
        manifest,
        published_revision="a" * 40,
        remote_manifest_path="humanoidarena-v1/common/seed-0/hrvla-release-manifest.json",
    )
    validate_release_receipt(receipt, manifest)
    receipt["published_revision"] = "main"
    with pytest.raises(ValueError):
        validate_release_receipt(receipt, manifest)


def test_write_json_once_allows_only_identical_resume(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    write_json_once(path, {"value": 1})
    write_json_once(path, {"value": 1})
    with pytest.raises(FileExistsError):
        write_json_once(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}


def test_release_manifest_rejects_external_symlink(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
    outside.write_bytes(b"secret")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "link.bin").symlink_to(outside)
    try:
        with pytest.raises(ValueError, match="symlink escapes"):
            build_release_manifest(
                tmp_path,
                artifact_id="bad",
                artifact_type="dataset",
                root="artifact",
                repo_id="owner/repo",
                repo_type="dataset",
                path_in_repo="bad",
                source_revision=REVISION,
                provenance={},
            )
    finally:
        outside.unlink()
