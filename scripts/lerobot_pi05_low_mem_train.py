#!/usr/bin/env python3
"""Run pinned LeRobot training with transient BF16 PI0.5 initialization.

LeRobot's PI05Policy constructs the multi-billion-parameter model in the
process-wide default FP32 dtype, only then casts most weights to BF16. That
transient FP32 peak was OOM-killed on the 31 GiB HELIOS host before CUDA was
used. The checkpoint supplies every parameter, so its loaded values (not the
initial random values) determine the policy. This changes only construction
dtype; it restores the default before training begins.
"""

from __future__ import annotations

import functools
from pathlib import Path
import runpy
import sys

import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: lerobot_pi05_low_mem_train.py LEROBOT_TRAIN.py [options]")
    upstream = Path(sys.argv[1]).resolve(strict=True)
    original = PI05Policy.__init__
    original_load = PI05Policy.load_state_dict
    original_from_pretrained = PI05Policy.from_pretrained.__func__

    @functools.wraps(original)
    def initialize_bf16(self, *args, **kwargs):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            return original(self, *args, **kwargs)
        finally:
            torch.set_default_dtype(previous)

    PI05Policy.__init__ = initialize_bf16

    @functools.wraps(original_load)
    def record_complete_load(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("PI0.5 checkpoint did not load every model key")
        self._hrvla_loaded_full_checkpoint = True
        return result

    @classmethod
    def verified_from_pretrained(cls, *args, **kwargs):
        model = original_from_pretrained(cls, *args, **kwargs)
        # LeRobot's PI0.5 loader catches weight-loading errors and can return
        # random initialization. That must never become a benchmark candidate.
        if not getattr(model, "_hrvla_loaded_full_checkpoint", False):
            raise RuntimeError("PI0.5 pretrained weights were not loaded completely")
        return model

    PI05Policy.load_state_dict = record_complete_load
    PI05Policy.from_pretrained = verified_from_pretrained
    sys.argv = [str(upstream), *sys.argv[2:]]
    print("HRVLA: BF16-only PI0.5 construction; standard dtype restored before training", flush=True)
    runpy.run_path(str(upstream), run_name="__main__")


if __name__ == "__main__":
    main()
