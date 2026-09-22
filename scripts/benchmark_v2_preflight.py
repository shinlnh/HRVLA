#!/usr/bin/env python3
"""Fail-closed readiness audit for one architecture × public-suite branch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_STATUS = {"pending", "ready", "complete"}


def _inside(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or not relative:
        raise ValueError("lock paths must be nonempty and relative to the checkout")
    resolved = (root / path).resolve()
    if resolved != root.resolve() and root.resolve() not in resolved.parents:
        raise ValueError("lock path escapes the checkout")
    return resolved


def _check_lock(root: Path, name: str, lock: Any, errors: list[str]) -> None:
    if not isinstance(lock, dict):
        errors.append(f"{name}: missing immutable file lock")
        return
    if not isinstance(lock.get("sha256"), str) or not SHA256.fullmatch(lock["sha256"]):
        errors.append(f"{name}: missing SHA-256")
        return
    try:
        path = _inside(root, lock.get("path", ""))
    except (TypeError, ValueError) as exc:
        errors.append(f"{name}: {exc}")
        return
    if not path.is_file():
        errors.append(f"{name}: locked file is missing")
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != lock["sha256"]:
        errors.append(f"{name}: SHA-256 mismatch")


def inspect_contract(
    contract: dict[str, Any],
    repo_root: Path,
    *,
    branch: str,
    architecture_head: str | None,
    architecture_in_history: bool,
    suite_head: str | None,
) -> list[str]:
    """Return every missing gate; an empty list alone permits a `ready` run."""

    errors: list[str] = []
    token, suite = contract.get("architecture_token"), contract.get("suite")
    if contract.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not isinstance(token, str) or suite not in {"HA", "SP"}:
        errors.append("architecture token or suite is invalid")
    elif branch != f"feat({token}-benchmark-{suite})/evaluate":
        errors.append("checked-out branch does not match the contract")
    revision = contract.get("architecture_revision")
    if not isinstance(revision, str) or not SHA40.fullmatch(revision):
        errors.append("architecture revision is not a full commit SHA")
    elif architecture_head != revision or not architecture_in_history:
        errors.append("architecture revision is stale or absent from branch history")
    suite_revision = contract.get("suite_revision")
    if not isinstance(suite_revision, str) or not SHA40.fullmatch(suite_revision):
        errors.append("public suite revision is not frozen")
    elif suite_head != suite_revision:
        errors.append("public suite checkout differs from frozen revision")
    for name in ("checkpoint_lock", "sonic_lock", "protocol_lock", "seed_plan_lock"):
        _check_lock(repo_root, name, contract.get(name), errors)
    if contract.get("precision") not in {"bf16", "fp16", "cuda_int8", "cpu_int8"}:
        errors.append("policy precision/backend is not frozen")
    expected = contract.get("expected_episodes")
    if type(expected) is not int or expected < 1:
        errors.append("expected episode count is not frozen")
    status = contract.get("status")
    if status not in ALLOWED_STATUS:
        errors.append("status must be pending, ready, or complete")
    elif status == "pending":
        errors.append("branch remains pending")
    if status == "complete":
        result = contract.get("result_manifest")
        try:
            path = _inside(repo_root, result)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("episodes_observed") != expected:
                errors.append("result manifest does not match expected episode count")
            if payload.get("audit_passed") is not True:
                errors.append("result manifest has not passed audit")
        except (TypeError, ValueError, OSError, json.JSONDecodeError):
            errors.append("complete branch lacks a valid result manifest")
    return errors


def _git(repo: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=root / "benchmark/v2_contract.json")
    parser.add_argument("--suite-root", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    branch = _git(root, "symbolic-ref", "--short", "HEAD") or "DETACHED"
    architecture = contract.get("architecture_branch")
    revision = contract.get("architecture_revision")
    head = None
    if isinstance(architecture, str):
        head = _git(root, "rev-parse", f"refs/remotes/origin/{architecture}")
        if head is None:
            head = _git(root, "rev-parse", architecture)
    ancestor = bool(
        isinstance(revision, str)
        and subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", revision, "HEAD"],
            capture_output=True, check=False,
        ).returncode == 0
    )
    errors = inspect_contract(
        contract, root, branch=branch, architecture_head=head,
        architecture_in_history=ancestor,
        suite_head=_git(args.suite_root, "rev-parse", "HEAD"),
    )
    print(json.dumps({"branch": branch, "ready": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
