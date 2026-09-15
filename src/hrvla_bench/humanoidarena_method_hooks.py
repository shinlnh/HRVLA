"""Non-invasive evaluator hooks for internal ST/STR language routing."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .humanoidarena_method_runtime import HumanoidArenaMethodRuntime
from .plan import canonical_sha256


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def install_method_hooks(
    module: Any,
    programs: dict[str, Any],
    *,
    method_id: str,
    task_id: str,
    output_dir: Path,
    implementation_revision: str,
    recovery_state: dict[str, Any] | None = None,
    recovery_scenario_id: str | None = None,
    failure_start: bool = False,
    per_episode_output: bool = False,
    shared_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Install one provider wrapper while retaining dynamic per-episode state."""

    context = shared_context if shared_context is not None else {}
    state: dict[str, Any] = {
        "runtime": None,
        "env": None,
        "task_success": False,
        "episode_seed": None,
        "episode_output_dir": None,
        "provider_hooked": False,
    }

    original_run_episode = module._run_episode_once

    def method_run_episode(*args, **kwargs):
        spec = kwargs.get("spec")
        if spec is None and len(args) > 6:
            spec = args[6]
        if not isinstance(spec, dict):
            raise RuntimeError("could not resolve upstream episode specification")
        episode_output = output_dir
        if per_episode_output:
            episode_output = output_dir / f"trial-{int(spec['episode_index']):04d}"
        state["episode_output_dir"] = episode_output
        context["episode_output_dir"] = episode_output
        try:
            return original_run_episode(*args, **kwargs)
        finally:
            state["episode_output_dir"] = None
            context["episode_output_dir"] = None

    module._run_episode_once = method_run_episode

    original_reset = module._reset_environment_for_episode

    def method_reset(env, env_cfg, episode_seed):
        result = original_reset(env, env_cfg, episode_seed)
        episode_output = state.get("episode_output_dir")
        if not isinstance(episode_output, Path):
            raise RuntimeError("method episode output directory is unresolved")
        trace_path = episode_output / "method-trace.jsonl"
        if trace_path.exists():
            raise RuntimeError(f"method output directory contains a stale trace: {trace_path}")
        control_dt = getattr(env, "step_dt", None)
        if control_dt is None:
            control_dt = float(env.physics_dt) * int(env_cfg.decimation)
        runtime = HumanoidArenaMethodRuntime(
            programs,
            method_id=method_id,
            task_id=task_id,
            control_dt_s=float(control_dt),
            seed=int(episode_seed),
        )
        runtime.reset(env)
        state.update(
            runtime=runtime,
            env=env,
            task_success=False,
            episode_seed=int(episode_seed),
        )
        _append_jsonl(
            trace_path,
            {
                "event": "method_reset",
                "method_id": method_id,
                "task_id": task_id,
                "episode_seed": int(episode_seed),
                "program_id": programs["program_id"],
                "program_sha256": canonical_sha256(programs),
                "implementation_revision": implementation_revision,
            },
        )
        return result

    module._reset_environment_for_episode = method_reset

    original_reward = module._extract_reward_info

    def method_reward(env):
        result = original_reward(env)
        state["task_success"] = float(result["raw_total"]) >= 1.0
        return result

    module._extract_reward_info = method_reward

    original_notify = module._notify_action_provider_env_reset

    def method_notify(provider):
        result = original_notify(provider)
        client = getattr(provider, "_lerobot_http_client", None)
        if client is not None:
            if state.get("episode_seed") is None:
                raise RuntimeError("method policy seed is unresolved at provider reset")
            client.reset(seed=int(state["episode_seed"]))
        if state["provider_hooked"]:
            return result
        original_fetch = provider._fetch_lerobot_action_chunk

        def method_fetch():
            runtime = state.get("runtime")
            env = state.get("env")
            episode_output = state.get("episode_output_dir")
            if runtime is None or env is None or not isinstance(episode_output, Path):
                raise RuntimeError("method runtime disappeared during action acquisition")
            recovery_active = bool(failure_start)
            if recovery_state is not None and not failure_start:
                recovery_runtime = recovery_state.get("runtime")
                recovery_active = bool(
                    recovery_runtime is not None and recovery_runtime.triggered
                )
            semantic_action = getattr(provider, "_latest_vla_action", None)
            decision = runtime.instruction(
                env,
                task_success=bool(state["task_success"]),
                semantic_action=semantic_action,
                recovery_scenario_id=recovery_scenario_id,
                recovery_active=recovery_active,
            )
            provider.task_name = decision.instruction
            _append_jsonl(
                episode_output / "method-trace.jsonl",
                {
                    "event": "instruction_selected",
                    "decision_index": runtime.decisions + runtime.recovery_decisions - 1,
                    "method_id": method_id,
                    "task_id": task_id,
                    "recovery_scenario_id": recovery_scenario_id,
                    "recovery_active": recovery_active,
                    **decision.to_dict(),
                },
            )
            return original_fetch()

        provider._fetch_lerobot_action_chunk = method_fetch
        state["provider_hooked"] = True
        return result

    module._notify_action_provider_env_reset = method_notify

    original_payload = module._build_result_payload

    def method_payload(*args, **kwargs):
        payload = original_payload(*args, **kwargs)
        runtime = state.get("runtime")
        episode_output = state.get("episode_output_dir")
        if runtime is None or not isinstance(episode_output, Path):
            raise RuntimeError("method runtime did not produce an episode summary")
        summary = {
            **runtime.summary(),
            "program_sha256": canonical_sha256(programs),
            "implementation_revision": implementation_revision,
            "trace_sha256": canonical_sha256(
                [
                    json.loads(line)
                    for line in (episode_output / "method-trace.jsonl").read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line
                ]
            ),
            "policy_reset_seed": int(state["episode_seed"]),
        }
        payload["hrvla_method"] = summary
        _write_json_atomic(episode_output / "method-summary.json", summary)
        return payload

    module._build_result_payload = method_payload
    return state


__all__ = ["install_method_hooks"]
