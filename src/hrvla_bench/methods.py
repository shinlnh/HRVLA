"""Validation for the immutable ST/STR/RT benchmark method registry."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Any


SHA40 = set("0123456789abcdef")
EXPECTED_FEATURES = {
    "gr00t_sonic": {"subtask": False, "recovery": False, "retrained": False},
    "gr00t_st": {"subtask": True, "recovery": False, "retrained": False},
    "gr00t_st_rt": {"subtask": True, "recovery": False, "retrained": True},
    "gr00t_str": {"subtask": True, "recovery": True, "retrained": False},
    "gr00t_str_rt": {"subtask": True, "recovery": True, "retrained": True},
}


def _sha40(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and not (set(value) - SHA40)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_method_registry(
    registry: dict[str, Any], artifact_lock: dict[str, Any], repo_root: Path
) -> None:
    """Reject ambiguous taxonomy, mutable refs, and unpublished project weights."""
    if registry.get("schema_version") != "1.0":
        raise ValueError("method registry schema_version must be '1.0'")
    lock_reference = registry.get("artifact_lock", {})
    lock_path = repo_root / str(lock_reference.get("path", ""))
    if not lock_path.is_file() or _file_sha256(lock_path) != lock_reference.get("sha256"):
        raise ValueError("artifact lock path or SHA-256 does not match the registry")
    if artifact_lock.get("status") != "published_verified":
        raise ValueError("artifact lock is not published_verified")
    if artifact_lock.get("publication_targets", {}).get("published") is not True:
        raise ValueError("project artifacts are not marked published")

    methods = registry.get("methods")
    if not isinstance(methods, list):
        raise ValueError("methods must be a list")
    by_id = {method.get("id"): method for method in methods}
    if set(by_id) != set(EXPECTED_FEATURES) or len(by_id) != len(methods):
        raise ValueError("method registry must contain each ST/STR/RT baseline exactly once")

    artifact_groups = artifact_lock.get("artifacts", {})
    for method_id, expected_features in EXPECTED_FEATURES.items():
        method = by_id[method_id]
        if method.get("features") != expected_features:
            raise ValueError(f"{method_id}: feature taxonomy does not match its suffix")
        revision = method.get("implementation_revision")
        if not _sha40(revision):
            raise ValueError(f"{method_id}: implementation_revision must be a Git SHA")
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise ValueError(f"{method_id}: implementation revision is unavailable")
        checkpoint = method.get("checkpoint", {})
        if not _sha40(checkpoint.get("revision")):
            raise ValueError(f"{method_id}: checkpoint revision must be immutable")
        if expected_features["retrained"]:
            group_name = checkpoint.get("artifact_group")
            group = artifact_groups.get(group_name)
            if not isinstance(group, dict):
                raise ValueError(f"{method_id}: artifact group is missing from the lock")
            if checkpoint.get("repo_id") != group.get("repo_id"):
                raise ValueError(f"{method_id}: checkpoint repository does not match the lock")
            if checkpoint.get("path") != group.get("path_in_repo"):
                raise ValueError(f"{method_id}: checkpoint path does not match the lock")
            if checkpoint.get("revision") != group.get("published_commit"):
                raise ValueError(f"{method_id}: checkpoint revision does not match publication")

    comparisons = registry.get("comparison_families")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("comparison_families must be a non-empty list")
    seen: set[str] = set()
    for comparison in comparisons:
        comparison_id = comparison.get("id")
        if not comparison_id or comparison_id in seen:
            raise ValueError("comparison family id is missing or duplicated")
        seen.add(comparison_id)
        method_ids = comparison.get("methods")
        if not isinstance(method_ids, list) or len(method_ids) != 2:
            raise ValueError(f"{comparison_id}: comparison must contain exactly two methods")
        if not set(method_ids).issubset(by_id):
            raise ValueError(f"{comparison_id}: comparison names an unknown method")
        if not comparison.get("controlled_difference"):
            raise ValueError(f"{comparison_id}: controlled_difference is required")
