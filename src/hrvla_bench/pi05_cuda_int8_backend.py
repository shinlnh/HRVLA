"""Full-CUDA INT8 weight-only backend for PI0.5 deployment.

Large Linear layers are quantized with bitsandbytes while the two skinny 40-D
action projections remain in their checkpoint BF16 dtype.  This keeps the full
policy resident on a 16 GiB GPU alongside Isaac Sim without using unsupported
small-shape INT8 GEMMs.
"""

from __future__ import annotations

import types
from typing import Any

import torch
from torch import nn


MIN_QUANTIZED_DIM = 256


def should_quantize_linear(module: nn.Linear) -> bool:
    """Return whether a Linear layer is large enough for the CUDA INT8 path."""

    return min(module.in_features, module.out_features) >= MIN_QUANTIZED_DIM


def _replace_linear_modules(
    module: nn.Module,
    device: torch.device,
    bnb: Any,
) -> tuple[int, int, int]:
    replaced = 0
    skipped = 0
    original_bytes = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if not should_quantize_linear(child):
                skipped += 1
                continue
            original_bytes += child.weight.numel() * child.weight.element_size()
            if child.bias is not None:
                original_bytes += child.bias.numel() * child.bias.element_size()
            weight = child.weight.detach().to(device="cpu", copy=True)
            bias = (
                None
                if child.bias is None
                else child.bias.detach().to(device="cpu", copy=True)
            )
            quantized = bnb.nn.Linear8bitLt(
                child.in_features,
                child.out_features,
                bias=bias is not None,
                has_fp16_weights=False,
                threshold=6.0,
            )
            quantized.weight = bnb.nn.Int8Params(
                weight,
                requires_grad=False,
                has_fp16_weights=False,
            )
            if bias is not None:
                quantized.bias = nn.Parameter(bias, requires_grad=False)
            quantized = quantized.to(device)
            setattr(module, name, quantized)
            del child, weight, bias
            torch.cuda.empty_cache()
            replaced += 1
        else:
            child_replaced, child_skipped, child_bytes = _replace_linear_modules(
                child, device, bnb
            )
            replaced += child_replaced
            skipped += child_skipped
            original_bytes += child_bytes
    return replaced, skipped, original_bytes


def configure_cuda_int8_weight_only(policy: Any, device_name: str) -> dict[str, Any]:
    """Quantize a loaded PI0.5 policy in-place and retain it fully on CUDA."""

    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError(f"cuda_int8_weight_only requires a CUDA device, got {device}")
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_int8_weight_only requested but CUDA is unavailable")
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise RuntimeError(
            "cuda_int8_weight_only requires bitsandbytes==0.50.2 in the policy venv"
        ) from exc
    if bnb.__version__ != "0.50.2":
        raise RuntimeError(
            f"cuda_int8_weight_only requires bitsandbytes==0.50.2, got {bnb.__version__}"
        )

    model = policy.model
    # PI0.5's checkpoint enables max-autotune compilation. bitsandbytes Linear
    # modules carry mutable quantization state and are not CUDA-graph safe across
    # repeated HTTP requests, so bind the original eager methods explicitly.
    model.sample_actions = types.MethodType(type(model).sample_actions, model)
    model.forward = types.MethodType(type(model).forward, model)
    model.config.compile_model = False

    replaced, skipped, original_bytes = _replace_linear_modules(model, device, bnb)
    policy.eval()
    torch.cuda.synchronize(device)
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in policy.parameters()
    )
    return {
        "backend": "cuda_int8_weight_only",
        "policy_device": str(device),
        "bitsandbytes_version": bnb.__version__,
        "quantized_linear_layers": replaced,
        "bf16_linear_layers": skipped,
        "minimum_quantized_dimension": MIN_QUANTIZED_DIM,
        "original_quantized_linear_bytes": original_bytes,
        "parameter_bytes": parameter_bytes,
        "compiled": False,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(device),
    }


__all__ = [
    "MIN_QUANTIZED_DIM",
    "configure_cuda_int8_weight_only",
    "should_quantize_linear",
]
