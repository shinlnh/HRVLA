from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hrvla_bench.pi05_cuda_int8_backend import (
    MIN_QUANTIZED_DIM,
    should_quantize_linear,
)
from hrvla_bench.pi05_hybrid_backend import CUDA_INT8_BACKEND, normalize_backend


def test_cuda_int8_backend_is_explicit() -> None:
    assert normalize_backend(" CUDA_INT8_WEIGHT_ONLY ") == CUDA_INT8_BACKEND


def test_large_linear_is_quantized() -> None:
    assert should_quantize_linear(torch.nn.Linear(1024, 2048))


@pytest.mark.parametrize("shape", [(40, 1024), (1024, 40), (128, 128)])
def test_skinny_linear_stays_in_checkpoint_precision(shape: tuple[int, int]) -> None:
    in_features, out_features = shape
    assert not should_quantize_linear(torch.nn.Linear(in_features, out_features))
    assert min(shape) < MIN_QUANTIZED_DIM
