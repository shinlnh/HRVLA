from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_humanoidarena_recovery_oracle.py"
SPEC = importlib.util.spec_from_file_location("recovery_oracle_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
SUITE = json.loads(
    (ROOT / "benchmark/suites/hrvla_recovery_v0.json").read_text(encoding="utf-8")
)


def test_oracle_runner_maps_all_locked_scenarios() -> None:
    rows = RUNNER._scenario_rows(SUITE)
    assert len(rows) == 9
    assert sum(row["protocol"] == "failure_start" for row in rows) == 3


def test_online_oracle_command_restores_then_injects(tmp_path: Path) -> None:
    row = next(
        row
        for row in RUNNER._scenario_rows(SUITE)
        if row["scenario_id"] == "support-state-push"
    )
    command = RUNNER.build_oracle_command(
        suite_path=ROOT / "benchmark/suites/hrvla_recovery_v0.json",
        row=row,
        attempt_dir=tmp_path / "attempt-0001",
        snapshot_path=tmp_path / "initial.json",
        snapshot_sha256="a" * 64,
        batch_path=tmp_path / "batch.json",
        port=18444,
    )
    assert "--per-episode-output" in command
    assert "--start-snapshot" in command
    assert "--restore-only" not in command
    assert command[command.index("--record_video_every_n") + 1] == "1"


def test_failure_start_oracle_command_never_reinjects(tmp_path: Path) -> None:
    row = next(
        row
        for row in RUNNER._scenario_rows(SUITE)
        if row["scenario_id"] == "box-drop-and-body-push"
    )
    command = RUNNER.build_oracle_command(
        suite_path=ROOT / "benchmark/suites/hrvla_recovery_v0.json",
        row=row,
        attempt_dir=tmp_path / "attempt-0001",
        snapshot_path=tmp_path / "failure.json",
        snapshot_sha256="b" * 64,
        batch_path=tmp_path / "batch.json",
        port=18444,
    )
    assert "--restore-only" in command
    assert "--capture-failure-snapshot" not in command
