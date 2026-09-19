from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from hrvla_bench.pi05_hybrid_backend import (
    CPU_BACKEND,
    HYBRID_BACKEND,
    _move_cache,
    normalize_backend,
)


def test_backend_is_cpu_by_default_and_hybrid_is_explicit() -> None:
    assert normalize_backend(None) == CPU_BACKEND
    assert normalize_backend("") == CPU_BACKEND
    assert normalize_backend(" HYBRID_CUDA_EXPERT ") == HYBRID_BACKEND


def test_unknown_backend_fails_instead_of_falling_back() -> None:
    with pytest.raises(ValueError, match="Unsupported PI0.5 backend"):
        normalize_backend("automatic")


def test_move_cache_copies_layers_and_preserves_source() -> None:
    source_tensor = torch.arange(4, dtype=torch.float32)
    cache = SimpleNamespace(
        layers=[SimpleNamespace(keys=source_tensor, values=source_tensor + 1)]
    )

    moved = _move_cache(cache, torch.device("cpu"))

    assert moved is not cache
    assert moved.layers[0] is not cache.layers[0]
    assert torch.equal(moved.layers[0].keys, source_tensor)
    assert torch.equal(moved.layers[0].values, source_tensor + 1)
