from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "run_humanoidarena_completion_supervisor_test",
        ROOT / "scripts/run_humanoidarena_completion_supervisor.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_stage_reuses_existing_partial_plan(tmp_path: Path) -> None:
    module = _module()
    plan = tmp_path / "release-plan.json"
    module.PRE_RECOVERY_RELEASE_PLAN = plan
    assert "--reuse-plan" not in module._release_stage_command("python")
    plan.write_text("{}\n", encoding="utf-8")
    command = module._release_stage_command("python")
    assert command[-2:] == ["--reuse-plan", str(plan)]
