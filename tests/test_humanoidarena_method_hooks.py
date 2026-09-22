from __future__ import annotations

import json
from pathlib import Path
import types

from hrvla_bench.humanoidarena_method_hooks import install_method_hooks
from hrvla_bench.humanoidarena_method_runtime import load_method_programs


ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = load_method_programs(ROOT / "benchmark/humanoidarena_method_programs.json")


class _Asset:
    def __init__(self) -> None:
        pose = [[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]]
        self.data = types.SimpleNamespace(
            root_link_pose_w=pose,
            root_state_w=[pose[0] + [0.0] * 6],
            root_vel_w=[[0.0] * 6],
        )


class _Env:
    step_dt = 0.02

    def __init__(self) -> None:
        self.scene = {"robot": _Asset(), "box": _Asset()}


class _Provider:
    def __init__(self) -> None:
        self.task_name = "original"
        self._latest_vla_action = None
        self.instructions = []
        self._lerobot_http_client = types.SimpleNamespace(reset=self._reset_client)
        self.policy_reset_seeds = []

    def _reset_client(self, seed=None):
        self.policy_reset_seeds.append(seed)

    def _fetch_lerobot_action_chunk(self):
        self.instructions.append(self.task_name)
        return [[0.0] * 40]


def _module(provider: _Provider, env: _Env, *, fetches: int = 1):
    module = types.SimpleNamespace()
    module._reset_environment_for_episode = lambda _env, _cfg, _seed: None
    module._extract_reward_info = lambda _env: {"raw_total": 0.0}
    module._notify_action_provider_env_reset = lambda _provider: None
    module._build_result_payload = lambda *_args, **_kwargs: {
        "episode_seed": 17,
        "success": False,
        "failure_reason": "timeout",
    }

    def run_episode(*_args, **_kwargs):
        module._reset_environment_for_episode(
            env, types.SimpleNamespace(decimation=1), 17
        )
        module._notify_action_provider_env_reset(provider)
        for index in range(fetches):
            if index == 1:
                provider._latest_vla_action = [0.0] * 38 + [1.0, 1.0]
            provider._fetch_lerobot_action_chunk()
        return module._build_result_payload()

    module._run_episode_once = run_episode
    return module


def test_hook_routes_st_prompts_and_embeds_audited_summary(tmp_path: Path) -> None:
    provider = _Provider()
    env = _Env()
    module = _module(provider, env, fetches=2)
    install_method_hooks(
        module,
        PROGRAMS,
        method_id="gr00t_st",
        task_id="pick_and_place_box",
        output_dir=tmp_path,
        implementation_revision="a" * 40,
    )
    result = module._run_episode_once(spec={"episode_index": 0})
    assert "Approach the box" in provider.instructions[0]
    assert "Lift the grasped box" in provider.instructions[1]
    assert result["hrvla_method"]["completed_transition_count"] == 1
    assert result["hrvla_method"]["policy_requests"] == 2
    assert result["hrvla_method"]["mean_policy_request_latency_ms"] >= 0.0
    assert result["hrvla_method"]["planner_calls"] == 2
    assert provider.policy_reset_seeds == [17]
    assert result["hrvla_method"]["policy_reset_seed"] == 17
    trace = [json.loads(line) for line in (tmp_path / "method-trace.jsonl").read_text().splitlines()]
    assert [row["event"] for row in trace] == [
        "method_reset",
        "instruction_selected",
        "instruction_selected",
    ]


def test_hook_routes_only_str_to_recovery_after_runtime_trigger(tmp_path: Path) -> None:
    provider = _Provider()
    env = _Env()
    module = _module(provider, env)
    recovery_state = {"runtime": types.SimpleNamespace(triggered=True, control_step=0)}
    install_method_hooks(
        module,
        PROGRAMS,
        method_id="gr00t_str",
        task_id="pick_and_place_box",
        output_dir=tmp_path,
        implementation_revision="a" * 40,
        recovery_state=recovery_state,
        recovery_scenario_id="box-missed-grasp-retry",
    )
    result = module._run_episode_once(spec={"episode_index": 0})
    assert "re-align" in provider.instructions[0]
    assert result["hrvla_method"]["recovery_decisions"] == 1
    assert result["hrvla_method"]["first_recovery_decision_control_step"] == 0


def test_pi05_st_hook_routes_real_provider_calls_and_checks_action40(tmp_path: Path) -> None:
    provider = _Provider()
    module = _module(provider, _Env(), fetches=2)
    install_method_hooks(
        module, PROGRAMS, method_id="pi05_st", task_id="pick_and_place_box",
        output_dir=tmp_path, implementation_revision="b" * 40,
    )
    result = module._run_episode_once(spec={"episode_index": 0})
    assert len(provider.instructions) == 2
    assert "Approach the box" in provider.instructions[0]
    assert "Lift the grasped box" in provider.instructions[1]
    assert result["hrvla_method"]["completed_transition_count"] == 1
    trace = [json.loads(line) for line in (tmp_path / "method-trace.jsonl").read_text().splitlines()]
    assert [row["policy_action_dim"] for row in trace[1:]] == [40, 40]


def test_pi05_st_hook_rejects_non_action40_chunk(tmp_path: Path) -> None:
    provider = _Provider()
    provider._fetch_lerobot_action_chunk = lambda: [[0.0] * 64]
    module = _module(provider, _Env())
    install_method_hooks(
        module, PROGRAMS, method_id="pi05_st", task_id="pick_and_place_box",
        output_dir=tmp_path, implementation_revision="b" * 40,
    )
    try:
        module._run_episode_once(spec={"episode_index": 0})
    except RuntimeError as error:
        assert "non-action40 chunk" in str(error)
    else:
        raise AssertionError("PI0.5 ST accepted a 64-D policy action")


def test_pi05_st_ignores_isaac_startup_reset_before_episode(tmp_path: Path) -> None:
    provider = _Provider()
    env = _Env()
    module = _module(provider, env)
    install_method_hooks(
        module, PROGRAMS, method_id="pi05_st", task_id="pick_and_place_box",
        output_dir=tmp_path, implementation_revision="b" * 40,
    )
    module._reset_environment_for_episode(env, types.SimpleNamespace(decimation=1), 17)
    assert not (tmp_path / "method-trace.jsonl").exists()
    result = module._run_episode_once(spec={"episode_index": 0})
    assert result["hrvla_method"]["policy_requests"] == 1
