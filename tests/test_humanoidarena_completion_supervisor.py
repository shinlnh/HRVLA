import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_humanoidarena_completion_supervisor.py"
SPEC = importlib.util.spec_from_file_location("completion_supervisor", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUPERVISOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUPERVISOR)


def test_hidden_audit_contains_all_five_registered_methods() -> None:
    command = SUPERVISOR._audit_hidden_command("python")
    records = [value for value in command if value.endswith("/records.jsonl")]
    assert len(records) == 5
    assert any("gr00t_sonic" in value for value in records)
    assert any("gr00t_str_rt" in value for value in records)


def test_post_external_gate_requires_every_independent_stage(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "status.json"
    required = {
        name: {"status": "complete"}
        for name in (
            "external_complete_audit", "common_gr00t_training", "common_gr00t_validation",
            "common_gr00t_hidden", "subtask_video_adjudication", "recovery_runtime_admission",
            "recovery_capture_manifest", "recovery_oracle", "admitted_suite",
            "recovery_training_dataset",
        )
    }
    path.write_text(
        __import__("json").dumps(
            {"status": "awaiting_audited_lock_freeze_and_rt_training", "stages": required}
        )
    )
    monkeypatch.setattr(SUPERVISOR, "POST_STATUS", path)
    assert SUPERVISOR._post_external_ready()
    required["recovery_oracle"]["status"] = "failed"
    path.write_text(__import__("json").dumps({"status": "awaiting_audited_lock_freeze_and_rt_training", "stages": required}))
    assert not SUPERVISOR._post_external_ready()


def test_dataset_family_frozen_requires_hash_bound_materialized_splits(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "dataset"
    manifests = {}
    for split in ("train", "validation"):
        (dataset / split).mkdir(parents=True)
        core = {"schema_version": 1, "split": split, "episodes": 3}
        digest = SUPERVISOR._canonical_sha256(core)
        manifests[split] = digest
        path = dataset / "manifests" / f"{split}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**core, "manifest_sha256": digest}))
    lock = tmp_path / "rt-lock.json"
    lock.write_text(
        json.dumps(
            {
                "subtask_rt_dataset": {
                    "path": "dataset",
                    "splits_materialized_before_selection": ["train", "validation"],
                    "manifests": {**manifests, "heldout": None},
                }
            }
        )
    )
    monkeypatch.setattr(SUPERVISOR, "ROOT", tmp_path)
    monkeypatch.setattr(SUPERVISOR, "RT_LOCK", lock)
    assert SUPERVISOR._dataset_family_frozen("subtask_rt_dataset")

    manifest = json.loads((dataset / "manifests/train.json").read_text())
    manifest["episodes"] = 4
    (dataset / "manifests/train.json").write_text(json.dumps(manifest))
    assert not SUPERVISOR._dataset_family_frozen("subtask_rt_dataset")


def test_record_existing_stage_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    status_path = tmp_path / "status.json"
    state = {"status": "waiting", "stages": {}}
    monkeypatch.setattr(SUPERVISOR, "_revision", lambda: "abc123")
    SUPERVISOR._record_existing_stage(state, status_path, "dataset", "already frozen")
    first = status_path.read_text()
    SUPERVISOR._record_existing_stage(state, status_path, "dataset", "different reason")
    assert status_path.read_text() == first
    assert state["stages"]["dataset"]["satisfied_by_existing_artifact"] is True
