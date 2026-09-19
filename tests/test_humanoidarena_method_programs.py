from __future__ import annotations

from pathlib import Path

from hrvla_bench.humanoidarena_method_runtime import load_method_programs


ROOT = Path(__file__).resolve().parents[1]


def test_method_programs_lock_seven_tasks_and_nine_recoveries() -> None:
    programs = load_method_programs(
        ROOT / "benchmark/humanoidarena_method_programs.json"
    )
    assert len(programs["tasks"]) == 7
    assert len(programs["recovery_instructions"]) == 9
    assert programs["planner"]["method"] == "adaptive_ttc"
