import json
from pathlib import Path

import pytest

from hrvla_bench.plan import canonical_sha256
from hrvla_bench.rt_lock_freeze import freeze_adjudication, freeze_dataset_manifests


ROOT = Path(__file__).resolve().parents[1]


def _lock() -> dict:
    return json.loads((ROOT / "config/humanoidarena-rt-training.lock.json").read_text())


def _adjudication(lock: dict) -> dict:
    policy = lock["subtask_rt_dataset"]["temporal_video_adjudication"]
    core = {
        "schema_version": 1,
        "status": "pass",
        "complete": True,
        "source_manifest_sha256": lock["source_dataset"]["manifest_sha256"],
        "method_program_sha256": lock["method_program_sha256"],
        "model_repo_id": policy["model_repo_id"],
        "model_revision": policy["model_revision"],
        "policy": {key: policy[key] for key in (
            "eligible_tasks", "sample_frames_per_variant", "sampling_variants",
            "minimum_consensus_variants",
            "minimum_boundary_confidence", "maximum_boundary_spread_fraction",
        )},
        "residual_fallback_rates": {"train": 0.01, "validation": 0.02},
        "records": [{"split": "train"}, {"split": "validation"}],
    }
    return {**core, "audit_sha256": canonical_sha256(core)}


def _manifest(lock: dict, family: str, split: str) -> dict:
    if family == "subtask_rt_dataset":
        core = {
            "split": split,
            "episodes": {"train": 490, "validation": 70}[split],
            "source_manifest_sha256": lock["source_dataset"]["manifest_sha256"],
            "method_program_sha256": lock["method_program_sha256"],
            "episode_fallback_rate": 0.01,
        }
    else:
        core = {
            "split": split,
            "episodes": {"train": 126, "validation": 27}[split],
            "scenarios": 9,
        }
    return {**core, "manifest_sha256": canonical_sha256(core)}


def test_freeze_adjudication_and_datasets_is_fail_closed() -> None:
    lock = _lock()
    adjudicated = freeze_adjudication(lock, _adjudication(lock))
    assert adjudicated["status"].startswith("adjudication_frozen")
    assert adjudicated["subtask_rt_dataset"]["temporal_video_adjudication"]["status"] == "ready"
    manifests = {
        family: {split: _manifest(adjudicated, family, split) for split in ("train", "validation")}
        for family in ("subtask_rt_dataset", "recovery_rt_dataset")
    }
    ready = freeze_dataset_manifests(adjudicated, manifests)
    assert ready["status"] == "ready_for_rt_training"
    assert ready["subtask_rt_dataset"]["manifests"]["heldout"] is None
    assert ready["recovery_rt_dataset"]["manifests"]["validation"] == manifests["recovery_rt_dataset"]["validation"]["manifest_sha256"]


def test_freeze_rejects_threshold_drift_and_hidden_access() -> None:
    lock = _lock()
    report = _adjudication(lock)
    report["policy"]["minimum_boundary_confidence"] = 0.1
    core = {key: value for key, value in report.items() if key != "audit_sha256"}
    report["audit_sha256"] = canonical_sha256(core)
    with pytest.raises(ValueError, match="policy differs"):
        freeze_adjudication(lock, report)
    report = _adjudication(lock)
    report["records"].append({"split": "heldout"})
    core = {key: value for key, value in report.items() if key != "audit_sha256"}
    report["audit_sha256"] = canonical_sha256(core)
    with pytest.raises(ValueError, match="locked split"):
        freeze_adjudication(lock, report)
