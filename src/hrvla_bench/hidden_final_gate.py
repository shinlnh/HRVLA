"""One-way validation freeze required before hidden-final benchmark access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .internal_matrix import audit_internal_records, validate_ready_checkpoint_lock
from .internal_protocol import INTERNAL_METHODS, validate_internal_protocol_lock
from .plan import canonical_sha256


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _validate_plan(plan: dict[str, Any], protocol: dict[str, Any]) -> None:
    core = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if canonical_sha256(core) != plan.get("plan_sha256"):
        raise ValueError("validation plan hash differs")
    if set(plan.get("methods", ())) != set(INTERNAL_METHODS):
        raise ValueError("validation plan method taxonomy differs")
    expected_seeds = set(protocol["splits"]["validation"]["rollout_seeds"])
    observed_seeds = {row["rollout_seed"] for row in plan.get("episodes", ())}
    if observed_seeds != expected_seeds:
        raise ValueError("gate input is not the frozen validation split")


def build_hidden_final_gate(
    *,
    validation_plan: dict[str, Any],
    protocol: dict[str, Any],
    checkpoint_lock: dict[str, Any],
    method_programs: dict[str, Any],
    validation_root: Path,
    source_revision_before_freeze: str,
) -> dict[str, Any]:
    """Bind complete validation records while making no outcome-dependent choice."""

    validate_internal_protocol_lock(protocol)
    _validate_plan(validation_plan, protocol)
    program_sha = canonical_sha256(method_programs)
    protocol_sha = canonical_sha256(protocol)
    validate_ready_checkpoint_lock(
        checkpoint_lock,
        method_program_sha256=program_sha,
        internal_protocol_sha256=protocol_sha,
    )
    if len(source_revision_before_freeze) != 40 or set(source_revision_before_freeze) - set("0123456789abcdef"):
        raise ValueError("source revision before freeze must be a full Git SHA")
    methods = {}
    for method_id in INTERNAL_METHODS:
        method_root = validation_root / method_id
        records_path = method_root / "records.jsonl"
        audit_path = method_root / "audit.json"
        records = _load_jsonl(records_path)
        expected_audit = audit_internal_records(validation_plan, method_id, records)
        observed_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if observed_audit != expected_audit:
            raise ValueError(f"{method_id}: validation audit differs from raw records")
        methods[method_id] = {
            "episode_records": len(records),
            "audit_sha256": observed_audit["audit_sha256"],
            "records_sha256": file_sha256(records_path),
        }
    core = {
        "schema_version": 1,
        "status": "ready_for_one_shot_hidden_final",
        "claim_boundary": "Validation is complete; proceed without outcome-dependent method, checkpoint, threshold, prompt, controller, or protocol changes.",
        "decision": "proceed_unchanged",
        "validation_plan_sha256": validation_plan["plan_sha256"],
        "checkpoint_lock_sha256": canonical_sha256(checkpoint_lock),
        "method_program_sha256": program_sha,
        "internal_protocol_sha256": protocol_sha,
        "source_revision_before_freeze": source_revision_before_freeze,
        "methods": methods,
    }
    return {**core, "gate_sha256": canonical_sha256(core)}


def validate_hidden_final_gate(
    gate: dict[str, Any],
    *,
    validation_plan: dict[str, Any],
    protocol: dict[str, Any],
    checkpoint_lock: dict[str, Any],
    method_programs: dict[str, Any],
    validation_root: Path,
) -> None:
    if gate.get("status") != "ready_for_one_shot_hidden_final" or gate.get("decision") != "proceed_unchanged":
        raise ValueError("hidden-final gate has not frozen an unchanged validation decision")
    core = {key: value for key, value in gate.items() if key != "gate_sha256"}
    if canonical_sha256(core) != gate.get("gate_sha256"):
        raise ValueError("hidden-final gate hash differs")
    expected = build_hidden_final_gate(
        validation_plan=validation_plan,
        protocol=protocol,
        checkpoint_lock=checkpoint_lock,
        method_programs=method_programs,
        validation_root=validation_root,
        source_revision_before_freeze=gate["source_revision_before_freeze"],
    )
    if expected != gate:
        raise ValueError("hidden-final gate differs from current validation evidence")


__all__ = ["build_hidden_final_gate", "file_sha256", "validate_hidden_final_gate"]
