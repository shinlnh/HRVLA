#!/usr/bin/env python3
"""Run HumanoidArena's VLA HTTP server with bounded-memory PI0.5 loading.

The upstream PI0.5 loader briefly holds a full float32 model and the checkpoint
state dict at the same time.  This wrapper keeps the locked upstream checkout
unchanged, constructs the module on PyTorch's ``meta`` device, and assigns the
exact safetensors weights directly onto the requested runtime device.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
HUMANOIDARENA_ROOT = ROOT / "_vendor" / "HumanoidArena"
LEROBOT_ROOT = HUMANOIDARENA_ROOT / "lerobot"
UPSTREAM_SERVER = LEROBOT_ROOT / "scripts" / "serve_lerobot_vla_http.py"

sys.path.insert(0, str(LEROBOT_ROOT / "src"))
spec = importlib.util.spec_from_file_location("humanoidarena_vla_server", UPSTREAM_SERVER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load upstream server: {UPSTREAM_SERVER}")
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)


def _configure_cpu_runtime() -> None:
    """Apply explicit thread settings before the first policy operation."""

    requested_threads = int(os.environ.get("OMP_NUM_THREADS", torch.get_num_threads()))
    requested_interop = int(os.environ.get("HRVLA_TORCH_INTEROP_THREADS", "2"))
    torch.set_num_threads(requested_threads)
    torch.set_num_interop_threads(requested_interop)
    print(
        "[hrvla_low_memory_loader] cpu_runtime "
        f"intraop_threads={torch.get_num_threads()} "
        f"interop_threads={torch.get_num_interop_threads()} "
        f"mkldnn={torch.backends.mkldnn.enabled}",
        flush=True,
    )


_configure_cpu_runtime()

if os.environ.get("HRVLA_VLA_DEBUG_LOGGING", "1").strip().lower() in {"0", "false", "no"}:
    upstream.LeRobotServerState._should_log_action_debug = lambda self: False


def _load_policy_low_memory(policy_dir: Path, device_name: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import _reconnect_relative_absolute_steps, get_policy_class
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    from lerobot.utils.control_utils import predict_action

    compat_dir_ctx, effective_policy_dir = upstream._prepare_compat_policy_dir(policy_dir)
    tokenizer_dir = ROOT / "_artifacts" / "HumanoidArena" / "tokenizers" / "paligemma-3b-pt-224"
    if not (tokenizer_dir / "tokenizer.json").is_file():
        raise FileNotFoundError(f"PaliGemma tokenizer is missing: {tokenizer_dir}")
    preprocessor_path = effective_policy_dir / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    preprocessor_payload = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    for step in preprocessor_payload.get("steps", []):
        registry_name = step.get("registry_name")
        if registry_name == "tokenizer_processor":
            step.setdefault("config", {})["tokenizer_name"] = str(tokenizer_dir)
        elif registry_name == "device_processor":
            step.setdefault("config", {})["device"] = str(torch.device(device_name))
    if preprocessor_path.is_symlink():
        preprocessor_path.unlink()
    preprocessor_path.write_text(
        json.dumps(preprocessor_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    config = PreTrainedConfig.from_pretrained(effective_policy_dir)
    if config.type != "pi05":
        raise ValueError(f"Low-memory loader only supports pi05, got {config.type!r}")

    target_device = str(torch.device(device_name))
    config.device = "meta"
    policy_cls = get_policy_class(config.type)
    with torch.device("meta"):
        policy = policy_cls(config)

    config.device = target_device
    model_file = effective_policy_dir / "model.safetensors"
    state_dict = load_file(str(model_file), device=target_device)
    fixed_state_dict = policy._fix_pytorch_state_dict_keys(state_dict, policy.config)
    remapped_state_dict = {
        key if key.startswith("model.") else f"model.{key}": value
        for key, value in fixed_state_dict.items()
    }
    missing_keys, unexpected_keys = policy.load_state_dict(
        remapped_state_dict,
        strict=True,
        assign=True,
    )
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Checkpoint key mismatch: missing={missing_keys}, unexpected={unexpected_keys}"
        )

    target = torch.device(target_device)
    for module in policy.modules():
        position_ids = getattr(module, "position_ids", None)
        if isinstance(position_ids, torch.Tensor) and position_ids.device.type == "meta":
            module.register_buffer(
                "position_ids",
                torch.arange(module.num_positions, device=target).expand((1, -1)),
                persistent=False,
            )
        inv_freq = getattr(module, "inv_freq", None)
        if isinstance(inv_freq, torch.Tensor) and inv_freq.device.type == "meta":
            fresh_rotary = type(module)(module.config, device=target)
            module.register_buffer("inv_freq", fresh_rotary.inv_freq, persistent=False)
            module.register_buffer(
                "original_inv_freq",
                fresh_rotary.original_inv_freq,
                persistent=False,
            )

    meta_tensors = [
        name
        for name, tensor in list(policy.named_parameters()) + list(policy.named_buffers())
        if tensor.device.type == "meta"
    ]
    if meta_tensors:
        raise RuntimeError(f"Unmaterialized checkpoint tensors: {meta_tensors[:10]}")

    policy.eval()
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in policy.parameters()
    )
    print(
        "[hrvla_low_memory_loader] loaded exact checkpoint "
        f"device={target_device} parameter_bytes={parameter_bytes}",
        flush=True,
    )

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        effective_policy_dir,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        effective_policy_dir,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    _reconnect_relative_absolute_steps(preprocessor, postprocessor)
    return (
        config,
        policy,
        preprocessor,
        postprocessor,
        predict_action,
        prepare_observation_for_inference,
        compat_dir_ctx,
    )


upstream._load_policy = _load_policy_low_memory


if __name__ == "__main__":
    upstream.main()
