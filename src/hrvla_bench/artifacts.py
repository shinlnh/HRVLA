"""Deterministic locks for locally produced benchmark checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


CHECKPOINT_GROUPS = {
    "GR00T-ST-RT": (
        "_artifacts/retraining/st-action-decoder-b4-ga4-s300/"
        "checkpoints/checkpoint-300"
    ),
    "GR00T-STR-RT": (
        "_artifacts/retraining/str-action-decoder-b16-ga1-s300/"
        "checkpoints/checkpoint-300"
    ),
    "arena-g1-subtask-split": "_artifacts/datasets/arena-g1-subtask-split",
    "arena-g1-recovery-split": "_artifacts/datasets/arena-g1-recovery-split",
}

PUBLICATION = {
    "GR00T-ST-RT": {
        "repo_type": "model",
        "repo_id": "shin0412/HRVLA",
        "path_in_repo": "checkpoints/GR00T-ST-RT/checkpoint-300",
        "published_commit": "f7ba25d113629e89c1e50757809d0f1e295bbeeb",
    },
    "GR00T-STR-RT": {
        "repo_type": "model",
        "repo_id": "shin0412/HRVLA",
        "path_in_repo": "checkpoints/GR00T-STR-RT/checkpoint-300",
        "published_commit": "5898d3177dc64322842fe1a42900cc9f952ea5e9",
    },
    "arena-g1-subtask-split": {
        "repo_type": "dataset",
        "repo_id": "shin0412/HRVLA",
        "path_in_repo": "arena-g1-subtask-split",
        "published_commit": "f6cc7b995ea214babc6e4f95c27202724e04df69",
    },
    "arena-g1-recovery-split": {
        "repo_type": "dataset",
        "repo_id": "shin0412/HRVLA",
        "path_in_repo": "arena-g1-recovery-split",
        "published_commit": "a745b9eba09f2beac02f6f018f3c9c369b0dc5a5",
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts


def build_artifact_lock(
    repo_root: Path,
    groups: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Hash every regular file in each selected final checkpoint directory."""
    selected = dict(groups or CHECKPOINT_GROUPS)
    artifacts: dict[str, Any] = {}
    for name, relative_root in sorted(selected.items()):
        if not _safe_relative_path(relative_root):
            raise ValueError(f"unsafe artifact root: {relative_root!r}")
        root = repo_root / relative_root
        if not root.is_dir():
            raise ValueError(f"artifact directory is missing: {root}")
        paths = sorted(path for path in root.rglob("*") if path.is_file())
        if not paths:
            raise ValueError(f"artifact directory contains no files: {root}")
        files = []
        for path in paths:
            relative = path.relative_to(repo_root).as_posix()
            files.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
        artifacts[name] = {
            "root": relative_root,
            "files": files,
            "total_bytes": sum(item["size_bytes"] for item in files),
            **PUBLICATION.get(name, {}),
        }

    return {
        "schema_version": "1.0",
        "status": "published_verified",
        "external_sources": {
            "base_vla": {
                "repo_id": "nvidia/GR00T-N1.7-3B",
                "revision": "2fc962b973bccdd5d8ce4f67cc63b264d6886495",
            },
            "proposal_model": {
                "repo_id": "nvidia/Cosmos-Reason2-2B",
                "revision": "9ce19a195e423419c349abfc86fd07178b230561",
            },
            "training_dataset": {
                "repo_id": "nvidia/Arena-G1-Static-PickNPlace-Task",
                "revision": "37ba80a99486a4c308477854f54b4e82e77777dc",
            },
        },
        "training_provenance": {
            "GR00T-ST-RT": "9dd8a7453d34c35123fe2bbe2e8affcd61034eb9",
            "GR00T-STR-RT": "3450540160dadfce60f2100904bdf6c1a0a929e5",
        },
        "publication_targets": {
            "model_repo": "shin0412/HRVLA",
            "model_revision": "5898d3177dc64322842fe1a42900cc9f952ea5e9",
            "dataset_repo": "shin0412/HRVLA",
            "dataset_revision": "a745b9eba09f2beac02f6f018f3c9c369b0dc5a5",
            "required": True,
            "published": True,
            "remote_checksum_verification": {
                "model_files": 32,
                "dataset_files": 858,
            },
        },
        "artifacts": artifacts,
    }


def verify_artifact_lock(lock: dict[str, Any], repo_root: Path) -> list[str]:
    """Return every local integrity error without stopping at the first one."""
    errors: list[str] = []
    if lock.get("schema_version") != "1.0":
        errors.append("schema_version must be '1.0'")
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return [*errors, "artifacts must be a non-empty object"]

    for name, group in sorted(artifacts.items()):
        files = group.get("files") if isinstance(group, dict) else None
        if not isinstance(files, list) or not files:
            errors.append(f"{name}: files must be a non-empty list")
            continue
        observed_total = 0
        seen: set[str] = set()
        for item in files:
            relative = item.get("path") if isinstance(item, dict) else None
            if not isinstance(relative, str) or not _safe_relative_path(relative):
                errors.append(f"{name}: unsafe artifact path {relative!r}")
                continue
            if relative in seen:
                errors.append(f"{name}: duplicate artifact path {relative}")
                continue
            seen.add(relative)
            path = repo_root / relative
            if not path.is_file():
                errors.append(f"{name}: missing {relative}")
                continue
            size = path.stat().st_size
            observed_total += size
            if size != item.get("size_bytes"):
                errors.append(f"{name}: size mismatch for {relative}")
                continue
            if file_sha256(path) != item.get("sha256"):
                errors.append(f"{name}: SHA-256 mismatch for {relative}")
        if observed_total != group.get("total_bytes"):
            errors.append(f"{name}: total_bytes mismatch")
    return errors
