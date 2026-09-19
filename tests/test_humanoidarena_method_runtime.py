from __future__ import annotations

from pathlib import Path
import types

import pytest


torch = pytest.importorskip("torch")

from hrvla_bench.humanoidarena_method_runtime import (
    HumanoidArenaMethodRuntime,
    load_method_programs,
)


ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = load_method_programs(ROOT / "benchmark/humanoidarena_method_programs.json")


class _Asset:
    def __init__(self) -> None:
        pose = torch.zeros((1, 7))
        pose[:, 3] = 1.0
        self.data = types.SimpleNamespace(
            root_link_pose_w=pose,
            root_state_w=torch.cat((pose, torch.zeros((1, 6))), dim=1),
            root_vel_w=torch.zeros((1, 6)),
        )


class _Env:
    def __init__(self) -> None:
        self.scene = {"robot": _Asset(), "box": _Asset()}


def test_base_row_keeps_the_whole_task_instruction() -> None:
    env = _Env()
    runtime = HumanoidArenaMethodRuntime(
        PROGRAMS,
        method_id="gr00t_sonic",
        task_id="pick_and_place_box",
        control_dt_s=0.02,
        seed=7,
    )
    runtime.reset(env)
    decision = runtime.instruction(env)
    assert decision.route == "whole_task"
    assert decision.instruction == "Move the box from the table onto the shelf."


def test_st_row_advances_only_after_source_observed_transition() -> None:
    env = _Env()
    runtime = HumanoidArenaMethodRuntime(
        PROGRAMS,
        method_id="gr00t_st",
        task_id="pick_and_place_box",
        control_dt_s=0.02,
        seed=7,
    )
    runtime.reset(env)
    first = runtime.instruction(env)
    assert first.selected_skill_id == "approach_and_grasp_box"
    close = [0.0] * 38 + [1.0, 1.0]
    second = runtime.instruction(env, semantic_action=close)
    assert second.transition_triggered is True
    assert second.selected_skill_id == "lift_and_transport_box"


def test_only_str_rows_route_an_active_failure_to_recovery() -> None:
    env = _Env()
    st = HumanoidArenaMethodRuntime(
        PROGRAMS,
        method_id="gr00t_st",
        task_id="pick_and_place_box",
        control_dt_s=0.02,
        seed=7,
    )
    str_runtime = HumanoidArenaMethodRuntime(
        PROGRAMS,
        method_id="gr00t_str",
        task_id="pick_and_place_box",
        control_dt_s=0.02,
        seed=7,
    )
    st.reset(env)
    str_runtime.reset(env)
    assert st.instruction(
        env, recovery_scenario_id="box-missed-grasp-retry", recovery_active=True
    ).route != "transition_recovery"
    recovery = str_runtime.instruction(
        env, recovery_scenario_id="box-missed-grasp-retry", recovery_active=True
    )
    assert recovery.route == "transition_recovery"
    assert "re-align" in recovery.instruction
