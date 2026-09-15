import importlib.util
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
