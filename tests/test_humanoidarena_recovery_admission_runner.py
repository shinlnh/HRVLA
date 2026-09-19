from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_humanoidarena_recovery_admission.py"
SPEC = importlib.util.spec_from_file_location("recovery_admission_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
SUITE = json.loads(
    (ROOT / "benchmark/suites/hrvla_recovery_v0.json").read_text(encoding="utf-8")
)


def test_task_rows_cover_all_nine_locked_scenarios() -> None:
    rows = RUNNER._task_rows(SUITE, set())
    assert len(rows) == 9
    assert len({row["scenario_id"] for row in rows}) == 9
    assert {row["task_key"] for row in rows} == {
        "boxing",
        "doubledesk",
        "football",
        "open_door",
        "pp_box",
        "sit_sofa",
        "vision_navi",
    }
    assert all(row["scenario"].get("boundary_driver") for row in rows)


def test_capture_horizon_is_bounded_by_evidence_not_task_rollout() -> None:
    rows = RUNNER._task_rows(SUITE, set())
    steps = {
        row["scenario_id"]: RUNNER._capture_max_steps(SUITE, row["scenario"])
        for row in rows
    }
    assert steps["box-drop-and-body-push"] == 60
    assert steps["sofa-approach-slip"] == 36
    assert steps["doorway-obstruction"] == 20
    assert steps["contact-recoil"] == 20
    assert max(steps.values()) == 60


def test_admission_is_fail_fast_by_default() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'parser.add_argument("--continue-on-error", action="store_true")' in source
    assert "if scenario_failed and not args.continue_on_error:" in source


def test_failure_start_command_captures_both_snapshots(tmp_path: Path) -> None:
    command = RUNNER.build_recovery_command(
        task_key="pp_box",
        scenario_id="box-drop-and-body-push",
        protocol="failure_start",
        output_dir=tmp_path / "runtime",
        batch_path=tmp_path / "batch.json",
        port=18444,
    )
    assert str(RUNNER.RECOVERY_RUNNER) in command
    assert "--capture-initial-snapshot" in command
    assert "--capture-failure-snapshot" in command
    assert "--record_video_every_n" in command
    assert command[command.index("--record_video_every_n") + 1] == "1"
    assert command[command.index("--episode_batch_json") + 1] == str(
        tmp_path / "batch.json"
    )


def test_online_command_does_not_capture_a_failure_snapshot(tmp_path: Path) -> None:
    command = RUNNER.build_recovery_command(
        task_key="football",
        scenario_id="support-state-push",
        protocol="online_failure",
        output_dir=tmp_path / "runtime",
        batch_path=tmp_path / "batch.json",
        port=18444,
    )
    assert "--capture-initial-snapshot" in command
    assert "--capture-failure-snapshot" not in command


def test_existing_capture_must_match_the_current_implementation(
    tmp_path: Path, monkeypatch
) -> None:
    attempt = tmp_path / "attempt-0001"
    attempt.mkdir()
    (attempt / "episode.json").write_text("{}")
    monkeypatch.setattr(
        RUNNER,
        "audit_recovery_runtime_trace",
        lambda *_args: {"audit_sha256": "a" * 64, "implementation_revision": "1" * 40},
    )
    assert (
        RUNNER._validated_existing(SUITE, "support-state-push", tmp_path, "2" * 40)
        is None
    )
    validated = RUNNER._validated_existing(
        SUITE, "support-state-push", tmp_path, "1" * 40
    )
    assert validated is not None
    assert validated[1] == attempt
