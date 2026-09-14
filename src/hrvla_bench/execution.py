"""Adapter contract and resumable execution for planned benchmark episodes."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
from typing import Any, Protocol

from .plan import canonical_sha256
from .score import validate_record


class EpisodeAdapter(Protocol):
    """Method-specific bridge to an Isaac Lab environment and policy."""

    def run_episode(self, episode: dict[str, Any]) -> dict[str, Any]:
        """Return outcome fields without changing any planned identifiers."""


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    method_id: str
    evaluation_track: str
    policy_checkpoint_id: str
    controller_id: str
    simulator_revision: str


def load_adapter(spec: str) -> EpisodeAdapter:
    """Load an adapter instance from ``package.module:attribute``."""
    if ":" not in spec:
        raise ValueError("adapter must use package.module:attribute syntax")
    module_name, attribute = spec.rsplit(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    adapter = factory() if callable(factory) else factory
    if not callable(getattr(adapter, "run_episode", None)):
        raise ValueError("adapter must provide run_episode(episode)")
    return adapter


def validate_plan_hash(plan: dict[str, Any]) -> None:
    claimed = plan.get("plan_sha256")
    core = {key: value for key, value in plan.items() if key != "plan_sha256"}
    actual = canonical_sha256(core)
    if claimed != actual:
        raise ValueError(f"plan hash mismatch: {claimed!r} != {actual}")


def run_plan(
    plan: dict[str, Any],
    identity: RunIdentity,
    adapter: EpisodeAdapter,
    output: Path,
    *,
    allow_draft: bool = False,
) -> list[dict[str, Any]]:
    """Run a frozen plan with crash-safe incremental progress and atomic publish."""
    validate_plan_hash(plan)
    if identity.method_id not in plan["methods"]:
        raise ValueError(f"method {identity.method_id!r} is not in the plan")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    records: list[dict[str, Any]] = []
    if partial.is_file():
        with partial.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    validate_record(record, allow_unverified=allow_draft)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"{partial}:{line_number}: {exc}") from exc
                records.append(record)
    if len(records) > len(plan["episodes"]):
        raise ValueError("partial output contains more records than the plan")

    for index, (record, episode) in enumerate(zip(records, plan["episodes"])):
        expected = {
            "run_id": identity.run_id,
            "episode_id": f"{identity.run_id}-{index:06d}",
            "method_id": identity.method_id,
            "evaluation_track": identity.evaluation_track,
            "suite_id": plan["suite_id"],
            "suite_sha256": plan["suite_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "task_id": episode["task_id"],
            "scenario_id": episode["scenario_id"],
            "protocol": episode["protocol"],
            "training_seed": episode["training_seed"],
            "rollout_seed": episode["rollout_seed"],
            "initial_snapshot_id": episode["initial_snapshot_id"],
            "failure_snapshot_id": episode.get("failure_snapshot_id"),
            "failure": episode.get("failure"),
            "policy_checkpoint_id": identity.policy_checkpoint_id,
            "controller_id": identity.controller_id,
            "simulator_revision": identity.simulator_revision,
        }
        mismatched = [key for key, value in expected.items() if record.get(key) != value]
        if mismatched:
            raise ValueError(
                f"partial output does not match plan at episode {index}: {mismatched}"
            )

    with partial.open("a" if records else "w", encoding="utf-8") as stream:
        for index, episode in enumerate(plan["episodes"][len(records) :], len(records)):
            if episode["admission_status"] != "admitted" and not allow_draft:
                raise ValueError(
                    f"refusing non-admitted scenario {episode['scenario_id']!r}; "
                    "admit it through the oracle protocol first"
                )
            outcome = adapter.run_episode(dict(episode))
            protected = {
                "run_id",
                "episode_id",
                "method_id",
                "evaluation_track",
                "suite_id",
                "suite_sha256",
                "plan_sha256",
                "task_id",
                "scenario_id",
                "protocol",
                "training_seed",
                "rollout_seed",
                "initial_snapshot_id",
                "failure_snapshot_id",
                "failure",
                "policy_checkpoint_id",
                "controller_id",
                "simulator_revision",
            }
            overlap = protected & set(outcome)
            if overlap:
                raise ValueError(
                    f"adapter attempted to override planned fields: {sorted(overlap)}"
                )
            failure = episode.get("failure")
            record = {
                "schema_version": "1.0",
                "run_id": identity.run_id,
                "episode_id": f"{identity.run_id}-{index:06d}",
                "method_id": identity.method_id,
                "evaluation_track": identity.evaluation_track,
                "suite_id": plan["suite_id"],
                "suite_sha256": plan["suite_sha256"],
                "plan_sha256": plan["plan_sha256"],
                "task_id": episode["task_id"],
                "scenario_id": episode["scenario_id"],
                "protocol": episode["protocol"],
                "training_seed": episode["training_seed"],
                "rollout_seed": episode["rollout_seed"],
                "initial_snapshot_id": episode["initial_snapshot_id"],
                "failure_snapshot_id": episode.get("failure_snapshot_id"),
                "policy_checkpoint_id": identity.policy_checkpoint_id,
                "controller_id": identity.controller_id,
                "simulator_revision": identity.simulator_revision,
                "failure": failure,
                **outcome,
            }
            validate_record(record, allow_unverified=allow_draft)
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            records.append(record)

    partial.replace(output)
    return records
