"""Content-addressed release manifests for HumanoidArena benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any, Iterable

from .plan import canonical_sha256


RUNTIME_SIGNATURE_PATHS = (
    "benchmark/humanoidarena_method_programs.json",
    "config/g1_humanoidarena_refpose_config.py",
    "scripts/run_humanoidarena_internal_episode.py",
    "scripts/run_humanoidarena_internal_matrix.py",
    "scripts/run_humanoidarena_recovery_episode.py",
    "scripts/serve_humanoidarena_gr00t.py",
    "src/hrvla_bench/humanoidarena_bridge.py",
    "src/hrvla_bench/humanoidarena_gr00t_http.py",
    "src/hrvla_bench/humanoidarena_method_hooks.py",
    "src/hrvla_bench/humanoidarena_method_runtime.py",
    "src/hrvla_bench/humanoidarena_recovery_runtime.py",
    "src/hrvla_bench/humanoidarena_recovery_signals.py",
    "src/hrvla_bench/internal_matrix.py",
    "src/hrvla_bench/isaac_events.py",
    "src/hrvla_bench/isaac_snapshot.py",
    "src/hrvla_bench/methods.py",
    "src/hrvla_bench/plan.py",
    "src/hrvla_bench/recovery_event_detector.py",
    "src/hrvla_bench/recovery_injector_contract.py",
    "src/hrvla_bench/recovery_restore.py",
    "src/hrvla_bench/recovery_runtime_audit.py",
    "src/hrvla_subtask/adapter.py",
    "src/hrvla_subtask/backends.py",
    "src/hrvla_subtask/model.py",
    "src/hrvla_subtask/planner.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _artifact_files(repo_root: Path, artifact_root: Path) -> list[Path]:
    paths = sorted(path for path in artifact_root.rglob("*") if path.is_file())
    if not paths:
        raise ValueError(f"artifact directory contains no files: {artifact_root}")
    escaped = [path for path in paths if not _inside(repo_root, path)]
    if escaped:
        raise ValueError(f"artifact symlink escapes repository: {escaped[0]}")
    return paths


def build_release_manifest(
    repo_root: Path,
    *,
    artifact_id: str,
    artifact_type: str,
    root: str,
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
    source_revision: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Hash one release directory without putting machine-specific paths in the lock."""

    if artifact_type not in {"checkpoint", "dataset"}:
        raise ValueError("artifact_type must be checkpoint or dataset")
    if repo_type not in {"model", "dataset"}:
        raise ValueError("repo_type must be model or dataset")
    if not all(_safe_relative(value) for value in (root, path_in_repo)):
        raise ValueError("artifact root and Hub path must be safe relative paths")
    if len(source_revision) != 40 or set(source_revision) - set("0123456789abcdef"):
        raise ValueError("source_revision must be a full Git SHA")
    artifact_root = repo_root / root
    if not artifact_root.is_dir() or not _inside(repo_root, artifact_root):
        raise ValueError(f"artifact root is missing or unsafe: {artifact_root}")
    files = []
    for path in _artifact_files(repo_root, artifact_root):
        relative = path.relative_to(artifact_root).as_posix()
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    core = {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "root": root,
        "repo_id": repo_id,
        "repo_type": repo_type,
        "path_in_repo": path_in_repo,
        "source_revision": source_revision,
        "provenance": provenance,
        "files": files,
        "total_bytes": sum(row["size_bytes"] for row in files),
    }
    return {**core, "manifest_sha256": canonical_sha256(core)}


def verify_release_manifest(manifest: dict[str, Any], repo_root: Path) -> list[str]:
    """Return all local integrity failures, including added or removed files."""

    errors = []
    expected_hash = manifest.get("manifest_sha256")
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if canonical_sha256(core) != expected_hash:
        errors.append("release manifest hash differs")
    root_value = manifest.get("root")
    if not isinstance(root_value, str) or not _safe_relative(root_value):
        return [*errors, "release root is unsafe"]
    artifact_root = repo_root / root_value
    if not artifact_root.is_dir() or not _inside(repo_root, artifact_root):
        return [*errors, "release root is missing or outside the repository"]
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        return [*errors, "release file list is empty"]
    expected_paths = set()
    observed_total = 0
    for row in rows:
        relative = row.get("path") if isinstance(row, dict) else None
        if not isinstance(relative, str) or not _safe_relative(relative):
            errors.append(f"unsafe release path: {relative!r}")
            continue
        if relative in expected_paths:
            errors.append(f"duplicate release path: {relative}")
            continue
        expected_paths.add(relative)
        path = artifact_root / relative
        if not path.is_file() or not _inside(repo_root, path):
            errors.append(f"missing or unsafe release file: {relative}")
            continue
        size = path.stat().st_size
        observed_total += size
        if size != row.get("size_bytes"):
            errors.append(f"release size differs: {relative}")
        elif file_sha256(path) != row.get("sha256"):
            errors.append(f"release SHA-256 differs: {relative}")
    actual_paths = {
        path.relative_to(artifact_root).as_posix()
        for path in _artifact_files(repo_root, artifact_root)
    }
    for relative in sorted(actual_paths - expected_paths):
        errors.append(f"unlocked release file: {relative}")
    for relative in sorted(expected_paths - actual_paths):
        errors.append(f"locked release file is absent: {relative}")
    if observed_total != manifest.get("total_bytes"):
        errors.append("release total_bytes differs")
    return errors


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    """Create an immutable JSON receipt, allowing an identical resume."""

    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to replace different frozen JSON: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def git_revision(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def runtime_signatures(repo_root: Path, paths: Iterable[str] = RUNTIME_SIGNATURE_PATHS) -> dict[str, str]:
    output = {}
    for relative in paths:
        if not _safe_relative(relative):
            raise ValueError(f"unsafe runtime signature path: {relative}")
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        output[relative] = file_sha256(path)
    return output


def release_receipt(
    manifest: dict[str, Any], *, published_revision: str, remote_manifest_path: str
) -> dict[str, Any]:
    if len(published_revision) != 40 or set(published_revision) - set("0123456789abcdef"):
        raise ValueError("published_revision must be a full immutable commit SHA")
    core = {
        "schema_version": 1,
        "artifact_id": manifest["artifact_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "repo_id": manifest["repo_id"],
        "repo_type": manifest["repo_type"],
        "path_in_repo": manifest["path_in_repo"],
        "published_revision": published_revision,
        "remote_manifest_path": remote_manifest_path,
        "remote_manifest_verified": True,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def validate_release_receipt(receipt: dict[str, Any], manifest: dict[str, Any]) -> None:
    core = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if canonical_sha256(core) != receipt.get("receipt_sha256"):
        raise ValueError("release receipt hash differs")
    for key in ("artifact_id", "manifest_sha256", "repo_id", "repo_type", "path_in_repo"):
        if receipt.get(key) != manifest.get(key):
            raise ValueError(f"release receipt {key} differs")
    revision = receipt.get("published_revision")
    if not isinstance(revision, str) or len(revision) != 40 or set(revision) - set("0123456789abcdef"):
        raise ValueError("release receipt revision is mutable")
    if receipt.get("remote_manifest_verified") is not True:
        raise ValueError("release receipt lacks remote manifest verification")


__all__ = [
    "RUNTIME_SIGNATURE_PATHS",
    "build_release_manifest",
    "file_sha256",
    "git_revision",
    "release_receipt",
    "runtime_signatures",
    "validate_release_receipt",
    "verify_release_manifest",
    "write_json_once",
]
