"""Opt-in heterogeneous CPU/CUDA inference for the PI0.5 policy.

The large vision-language prefix remains on CPU while the smaller action
expert and denoising projections run on CUDA.  The split is deliberately
explicit: callers must request the hybrid backend and failures never fall back
silently to CPU, which keeps benchmark provenance unambiguous.
"""

from __future__ import annotations

import copy
import types
from typing import Any

import torch

CPU_BACKEND = "cpu"
HYBRID_BACKEND = "hybrid_cuda_expert"
CUDA_INT8_BACKEND = "cuda_int8_weight_only"
SUPPORTED_BACKENDS = frozenset({CPU_BACKEND, HYBRID_BACKEND, CUDA_INT8_BACKEND})


def normalize_backend(value: str | None) -> str:
    backend = (value or CPU_BACKEND).strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_BACKENDS))
        raise ValueError(f"Unsupported PI0.5 backend {backend!r}; expected one of: {supported}")
    return backend


def _move_cache(cache: Any, device: torch.device) -> Any:
    """Copy a Transformers cache and move only its tensor payloads."""

    moved = copy.deepcopy(cache)
    layers = getattr(moved, "layers", None)
    if layers is not None:
        for layer in layers:
            for attr in ("keys", "values", "_sliding_window_tensor"):
                value = getattr(layer, attr, None)
                if isinstance(value, torch.Tensor):
                    setattr(layer, attr, value.to(device=device, non_blocking=False))
        return moved
    if isinstance(moved, torch.Tensor):
        return moved.to(device=device, non_blocking=False)
    if isinstance(moved, list):
        return [_move_cache(value, device) for value in moved]
    if isinstance(moved, tuple):
        return tuple(_move_cache(value, device) for value in moved)
    if isinstance(moved, dict):
        return {key: _move_cache(value, device) for key, value in moved.items()}
    return moved


@torch.compiler.disable
def _denoise_step_eager(model: Any, **kwargs: Any) -> torch.Tensor:
    """Keep CUDA expert numerics out of Inductor's fused approximation path."""

    return model.denoise_step(**kwargs)


@torch.no_grad()
def _sample_actions_hybrid(
    self,
    images,
    img_masks,
    tokens,
    masks,
    noise=None,
    num_steps=None,
    **kwargs,
):
    """Run the VLM prefix on CPU and all denoising steps on CUDA."""

    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    if self._rtc_enabled():
        raise RuntimeError("hybrid_cuda_expert does not support RTC-enabled PI0.5 policies")
    if num_steps is None:
        num_steps = self.config.num_inference_steps

    prefix_device = tokens.device
    expert_device = self._hrvla_expert_device
    if prefix_device.type != "cpu":
        raise RuntimeError(
            f"hybrid_cuda_expert requires a CPU prefix, got tokens on {prefix_device}"
        )

    batch_size = tokens.shape[0]
    if noise is None:
        actions_shape = (
            batch_size,
            self.config.chunk_size,
            self.config.max_action_dim,
        )
        # Draw on CPU before transfer so seeded CPU and hybrid runs start from
        # identical noise.  This makes numerical-equivalence gates meaningful.
        noise = self.sample_noise(actions_shape, prefix_device)

    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        images, img_masks, tokens, masks
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
    self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = (
        "eager"
    )
    _, past_key_values = self.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    prefix_pad_masks = prefix_pad_masks.to(device=expert_device, non_blocking=False)
    past_key_values = _move_cache(past_key_values, expert_device)
    x_t = noise.to(device=expert_device, non_blocking=False)
    dt = -1.0 / num_steps
    for step in range(num_steps):
        timestep = torch.full(
            (batch_size,),
            1.0 + step * dt,
            dtype=torch.float32,
            device=expert_device,
        )
        velocity = _denoise_step_eager(
            self,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=timestep,
        )
        x_t = x_t + dt * velocity

    return x_t.to(device=prefix_device, non_blocking=False)


def configure_hybrid_cuda_expert(policy: Any, expert_device_name: str) -> dict[str, Any]:
    """Move the PI0.5 action path to CUDA and install the split forward pass."""

    expert_device = torch.device(expert_device_name)
    if expert_device.type != "cuda":
        raise ValueError(f"PI0.5 expert device must be CUDA, got {expert_device}")
    if not torch.cuda.is_available():
        raise RuntimeError("hybrid_cuda_expert requested but CUDA is unavailable")

    model = policy.model
    modules = {
        "gemma_expert": model.paligemma_with_expert.gemma_expert,
        "action_in_proj": model.action_in_proj,
        "action_out_proj": model.action_out_proj,
        "time_mlp_in": model.time_mlp_in,
        "time_mlp_out": model.time_mlp_out,
    }
    for module in modules.values():
        module.to(expert_device)

    model._hrvla_expert_device = expert_device
    hybrid_sample_actions = types.MethodType(_sample_actions_hybrid, model)
    if model.config.compile_model:
        torch.set_float32_matmul_precision("high")
        hybrid_sample_actions = torch.compile(
            hybrid_sample_actions,
            mode=model.config.compile_mode,
        )
    model.sample_actions = hybrid_sample_actions

    expert_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for module in modules.values()
        for parameter in module.parameters()
    )
    torch.cuda.synchronize(expert_device)
    return {
        "backend": HYBRID_BACKEND,
        "prefix_device": "cpu",
        "expert_device": str(expert_device),
        "expert_parameter_bytes": expert_parameter_bytes,
        "compiled": bool(model.config.compile_model),
        "cuda_allocated_bytes": torch.cuda.memory_allocated(expert_device),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(expert_device),
    }
