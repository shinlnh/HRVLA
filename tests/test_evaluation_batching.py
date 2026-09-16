from __future__ import annotations

import time

import numpy as np
import torch

from hrvla_bench.evaluation_batching import (
    bounded_ordered_prefetch,
    seeded_noise_batch,
    stack_observations,
)


def _observation(value: int) -> dict:
    return {
        "video": {"camera": np.full((1, 2, 3, 4, 3), value, dtype=np.uint8)},
        "state": {"joints": np.full((1, 1, 2), value, dtype=np.float32)},
        "language": {"task": [[f"task-{value}"]]},
    }


def test_stack_observations_preserves_batch_order() -> None:
    stacked = stack_observations([_observation(2), _observation(7)])
    assert stacked["video"]["camera"].shape == (2, 2, 3, 4, 3)
    assert stacked["video"]["camera"][:, 0, 0, 0, 0].tolist() == [2, 7]
    assert stacked["state"]["joints"][:, 0, 0].tolist() == [2.0, 7.0]
    assert stacked["language"]["task"] == [["task-2"], ["task-7"]]


def test_seeded_noise_is_invariant_to_batch_neighbors() -> None:
    batched = seeded_noise_batch(
        [11, 29], sample_shape=(3, 4), dtype=torch.float32, device="cpu"
    )
    first = seeded_noise_batch(
        [11], sample_shape=(3, 4), dtype=torch.float32, device="cpu"
    )
    second = seeded_noise_batch(
        [29], sample_shape=(3, 4), dtype=torch.float32, device="cpu"
    )
    assert torch.equal(batched[0], first[0])
    assert torch.equal(batched[1], second[0])


def test_bounded_prefetch_keeps_input_order() -> None:
    def prepare(value: int) -> int:
        time.sleep(0.001 * (4 - value))
        return value * 2

    assert list(bounded_ordered_prefetch(range(4), prepare, workers=2)) == [0, 2, 4, 6]
