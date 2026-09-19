"""Deterministic helpers for CPU-prefetched, GPU-batched VLA evaluation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def stack_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack already-batched GR00T observations without changing sample order."""

    if not observations:
        raise ValueError("at least one observation is required")
    output: dict[str, Any] = {"video": {}, "state": {}, "language": {}}
    for modality in ("video", "state"):
        keys = tuple(observations[0][modality])
        if any(tuple(value[modality]) != keys for value in observations):
            raise ValueError(f"inconsistent {modality} keys in evaluation batch")
        for key in keys:
            output[modality][key] = np.concatenate(
                [value[modality][key] for value in observations], axis=0
            )
    language_keys = tuple(observations[0]["language"])
    if any(tuple(value["language"]) != language_keys for value in observations):
        raise ValueError("inconsistent language keys in evaluation batch")
    for key in language_keys:
        output["language"][key] = [
            item
            for observation in observations
            for item in observation["language"][key]
        ]
    return output


def seeded_noise_batch(
    seeds: list[int],
    *,
    sample_shape: tuple[int, ...],
    dtype: "torch.dtype",
    device: "torch.device | str",
    randn: Callable[..., "torch.Tensor"] | None = None,
) -> "torch.Tensor":
    """Create one independent RNG stream per sample for batch-size invariance."""

    import torch

    if not seeds:
        raise ValueError("at least one seed is required")
    if randn is None:
        randn = torch.randn
    samples = []
    for seed in seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        samples.append(
            randn(
                size=(1, *sample_shape),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )
    return torch.cat(samples, dim=0)


@contextmanager
def inject_seeded_action_noise(policy: Any, seeds: list[int]) -> Iterator[None]:
    """Replace exactly the GR00T action-head noise draw for a batched request.

    The locked upstream action head does not expose its initial diffusion noise.
    Intercepting the single, shape-checked draw lets batching preserve the exact
    per-request seed contract. Any upstream call-pattern drift fails closed.
    """

    import torch

    action_head = policy.model.action_head
    expected_shape = (
        len(seeds),
        int(action_head.config.action_horizon),
        int(action_head.action_dim),
    )
    original = torch.randn
    matched_calls = 0

    def replacement(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal matched_calls
        size = kwargs.get("size")
        if size is None and args:
            size = args[0] if len(args) == 1 else args
        if size is not None and tuple(size) == expected_shape:
            matched_calls += 1
            if matched_calls != 1:
                raise RuntimeError("GR00T action noise was requested more than once")
            if "dtype" not in kwargs or "device" not in kwargs:
                raise RuntimeError("GR00T action-noise draw omitted dtype or device")
            return seeded_noise_batch(
                seeds,
                sample_shape=expected_shape[1:],
                dtype=kwargs["dtype"],
                device=kwargs["device"],
                randn=original,
            )
        return original(*args, **kwargs)

    torch.randn = replacement
    try:
        yield
    finally:
        torch.randn = original
    if matched_calls != 1:
        raise RuntimeError(
            f"expected one GR00T action-noise draw, observed {matched_calls}"
        )


def bounded_ordered_prefetch(
    values: Iterable[int],
    function: Callable[[int], Any],
    *,
    workers: int,
    pending_per_worker: int = 2,
) -> Iterator[Any]:
    """Run CPU preparation concurrently with bounded memory and ordered output."""

    if workers < 1 or pending_per_worker < 1:
        raise ValueError("prefetch worker and pending counts must be positive")
    iterator = iter(values)
    pending: list[Future[Any]] = []
    limit = workers * pending_per_worker
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vla-prefetch") as pool:
        for _ in range(limit):
            try:
                pending.append(pool.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            future = pending.pop(0)
            yield future.result()
            try:
                pending.append(pool.submit(function, next(iterator)))
            except StopIteration:
                pass
