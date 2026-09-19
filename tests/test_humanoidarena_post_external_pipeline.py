from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_humanoidarena_post_external_pipeline.py"
SPEC = importlib.util.spec_from_file_location("humanoidarena_post_external", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


def test_gpu_first_stages_preserve_dependencies_and_put_tracked_output_last(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    status = {"stages": {}}

    def fake_stage(state, _status_path, name, command, *, dependencies=()):
        calls.append((name, command, dependencies))
        state["stages"][name] = {"status": "complete"}
        return True

    monkeypatch.setattr(PIPELINE, "_stage", fake_stage)
    assert PIPELINE._run_gpu_first_stages(status, tmp_path / "status.json", "py", "gpy")
    assert [name for name, _command, _dependencies in calls] == list(
        PIPELINE.GPU_FIRST_STAGE_NAMES
    )
    assert calls[1][2] == ("common_gr00t_training",)
    assert calls[2][2] == ("common_gr00t_validation",)
    assert calls[-1][0] == "subtask_video_adjudication"


def test_gpu_first_failure_is_reported_after_independent_stage_attempts(
    tmp_path: Path, monkeypatch
) -> None:
    status = {"stages": {}}

    def fake_stage(state, _status_path, name, _command, *, dependencies=()):
        complete = name != "common_gr00t_training" and not dependencies
        state["stages"][name] = {
            "status": "complete" if complete else "blocked_or_failed"
        }
        return complete

    monkeypatch.setattr(PIPELINE, "_stage", fake_stage)
    assert not PIPELINE._run_gpu_first_stages(
        status, tmp_path / "status.json", "py", "gpy"
    )
    assert status["stages"]["subtask_video_adjudication"]["status"] == "complete"
