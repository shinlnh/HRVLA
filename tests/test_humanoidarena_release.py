import json
import importlib.util
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


def _release_script():
    spec = importlib.util.spec_from_file_location(
        "release_humanoidarena_artifacts_test",
        Path(__file__).resolve().parents[1] / "scripts/release_humanoidarena_artifacts.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_partial_release_does_not_require_recovery_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    module = _release_script()
    module.ROOT = tmp_path
    common_lock = {
        "output_root": "common",
        "training_seeds": [0, 1, 2],
    }
    rt_lock = {
        "output_root": "rt",
        "training_seeds": [0, 1, 2],
        "methods": {"gr00t_st_rt": {"output_name": "st-rt"}},
        "subtask_rt_dataset": {"path": "st-data"},
    }
    (tmp_path / "common").mkdir()
    (tmp_path / "common/selection.json").write_text(
        json.dumps({"selection_sha256": "c" * 64}), encoding="utf-8"
    )
    (tmp_path / "rt/st-rt").mkdir(parents=True)
    (tmp_path / "rt/st-rt/selection.json").write_text(
        json.dumps({"selection_sha256": "s" * 64}), encoding="utf-8"
    )
    validated = []
    monkeypatch.setattr(module, "validate_selection", lambda selection, lock: 300)
    monkeypatch.setattr(
        module,
        "validate_rt_inputs",
        lambda root, rt, common, selection, methods: validated.append(methods),
    )
    monkeypatch.setattr(module, "_validate_rt_selection", lambda lock, method, selection: 300)
    monkeypatch.setattr(
        module,
        "_dataset_provenance",
        lambda lock, family: {"dataset_family": family, "manifests": {}},
    )
    monkeypatch.setattr(
        module,
        "seed_directory",
        lambda root, lock, seed: root / f"common/seed-{seed}",
    )
    monkeypatch.setattr(
        module,
        "run_directory",
        lambda root, lock, method, seed: root / f"rt/st-rt/seed-{seed}",
    )
    specs = module._artifact_specifications(
        common_lock, rt_lock, ("common", "subtask_rt")
    )
    assert validated == [("gr00t_st_rt",)]
    assert [row["artifact_id"] for row in specs] == [
        "common-seed-0",
        "common-seed-1",
        "common-seed-2",
        "subtask_rt-seed-0",
        "subtask_rt-seed-1",
        "subtask_rt-seed-2",
        "subtask-rt-dataset",
    ]


def test_stage_parser_accepts_reusable_partial_plans() -> None:
    args = _release_script().parser().parse_args(
        [
            "stage",
            "--family",
            "common",
            "--family",
            "subtask_rt",
            "--reuse-plan",
            "partial.json",
        ]
    )
    assert args.family == ["common", "subtask_rt"]
    assert args.reuse_plan == [Path("partial.json")]
