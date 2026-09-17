"""Mechanical freeze steps between observed RT artifacts and GR00T training."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .plan import canonical_sha256


def freeze_adjudication(
    lock: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    output = copy.deepcopy(lock)
    policy = output["subtask_rt_dataset"]["temporal_video_adjudication"]
    if policy.get("status") not in {"pending", "ready"}:
        raise ValueError("temporal adjudication lock is in an unknown state")
    core = {key: value for key, value in report.items() if key != "audit_sha256"}
    audit_sha = canonical_sha256(core)
    if report.get("audit_sha256") != audit_sha:
        raise ValueError("temporal adjudication report hash differs")
    if report.get("status") != "pass" or report.get("complete") is not True:
        raise ValueError("temporal adjudication did not pass completely")
    if report.get("source_manifest_sha256") != output["source_dataset"]["manifest_sha256"]:
        raise ValueError("temporal adjudication source dataset differs")
    if report.get("method_program_sha256") != output["method_program_sha256"]:
        raise ValueError("temporal adjudication method programs differ")
    for key in ("model_repo_id", "model_revision"):
        if report.get(key) != policy.get(key):
            raise ValueError(f"temporal adjudication {key} differs")
    expected_policy = {
        key: policy[key]
        for key in (
            "eligible_tasks",
            "sample_frames_per_variant",
            "sampling_variants",
            "minimum_consensus_variants",
            "minimum_boundary_confidence",
            "maximum_boundary_spread_fraction",
        )
    }
    if report.get("policy") != expected_policy:
        raise ValueError("temporal adjudication policy differs from the frozen thresholds")
    splits = set(output["subtask_rt_dataset"]["splits_materialized_before_selection"])
    residual = report.get("residual_fallback_rates", {})
    maximum = float(output["subtask_rt_dataset"]["maximum_episode_fallback_rate"])
    if set(residual) != splits or any(float(residual[split]) > maximum for split in splits):
        raise ValueError("temporal adjudication residual fallback gate failed")
    observed_splits = {str(row.get("split")) for row in report.get("records", [])}
    if not observed_splits <= splits:
        raise ValueError("temporal adjudication accessed a locked split")
    policy["status"] = "ready"
    policy["report_sha256"] = audit_sha
    output["status"] = "adjudication_frozen_waiting_for_dataset_materialization"
    return output


def _validate_manifest(
    lock: dict[str, Any], family: str, split: str, manifest: dict[str, Any]
) -> str:
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    digest = canonical_sha256(core)
    if manifest.get("manifest_sha256") != digest:
        raise ValueError(f"{family}/{split}: manifest hash differs")
    if manifest.get("split") != split:
        raise ValueError(f"{family}/{split}: split differs")
    if family == "subtask_rt_dataset":
        expected_episodes = {"train": 490, "validation": 70}[split]
        if manifest.get("episodes") != expected_episodes:
            raise ValueError(f"{family}/{split}: episode count differs")
        if manifest.get("source_manifest_sha256") != lock["source_dataset"]["manifest_sha256"]:
            raise ValueError(f"{family}/{split}: source dataset differs")
        if manifest.get("method_program_sha256") != lock["method_program_sha256"]:
            raise ValueError(f"{family}/{split}: method programs differ")
        if float(manifest.get("episode_fallback_rate", 1.0)) > float(
            lock["subtask_rt_dataset"]["maximum_episode_fallback_rate"]
        ):
            raise ValueError(f"{family}/{split}: fallback admission failed")
    else:
        expected_episodes = {"train": 126, "validation": 27}[split]
        if manifest.get("episodes") != expected_episodes or manifest.get("scenarios") != 9:
            raise ValueError(f"{family}/{split}: recovery matrix differs")
    return digest


def freeze_dataset_manifests(
    lock: dict[str, Any],
    manifests: dict[str, dict[str, dict[str, Any]]],
    families: tuple[str, ...] = ("subtask_rt_dataset", "recovery_rt_dataset"),
) -> dict[str, Any]:
    if lock["subtask_rt_dataset"]["temporal_video_adjudication"].get("status") != "ready":
        raise ValueError("adjudication must be frozen before dataset manifests")
    allowed = ("subtask_rt_dataset", "recovery_rt_dataset")
    if not families or len(set(families)) != len(families) or any(
        family not in allowed for family in families
    ):
        raise ValueError("dataset families must be a non-empty unique locked subset")
    output = copy.deepcopy(lock)
    for family in families:
        if set(manifests.get(family, {})) != {"train", "validation"}:
            raise ValueError(f"{family}: train and validation manifests are required")
        for split in ("train", "validation"):
            digest = _validate_manifest(output, family, split, manifests[family][split])
            output[family]["manifests"][split] = digest
        if output[family]["manifests"].get("heldout") is not None:
            raise ValueError(f"{family}: hidden manifest was accessed before selection")
    ready = {
        family: all(
            isinstance(output[family]["manifests"].get(split), str)
            for split in ("train", "validation")
        )
        for family in allowed
    }
    if all(ready.values()):
        output["status"] = "ready_for_rt_training"
    elif ready["subtask_rt_dataset"]:
        output["status"] = "subtask_rt_ready_waiting_for_recovery_dataset"
    elif ready["recovery_rt_dataset"]:
        output["status"] = "recovery_rt_ready_waiting_for_subtask_dataset"
    else:
        output["status"] = "adjudication_frozen_waiting_for_dataset_materialization"
    return output


def load_dataset_manifests(
    repo_root: Path,
    lock: dict[str, Any],
    families: tuple[str, ...] = ("subtask_rt_dataset", "recovery_rt_dataset"),
) -> dict[str, dict[str, dict[str, Any]]]:
    import json

    output = {}
    for family in families:
        root = repo_root / lock[family]["path"] / "manifests"
        output[family] = {
            split: json.loads((root / f"{split}.json").read_text(encoding="utf-8"))
            for split in ("train", "validation")
        }
    return output


__all__ = ["freeze_adjudication", "freeze_dataset_manifests", "load_dataset_manifests"]
