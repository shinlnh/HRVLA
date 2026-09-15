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
