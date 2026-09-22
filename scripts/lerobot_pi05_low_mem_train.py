#!/usr/bin/env python3
"""Run pinned LeRobot training with a strict, streaming PI0.5 weight loader.

The upstream PI0.5 loader first constructs 4.14B parameters in host memory,
then loads a full checkpoint into another host-memory allocation. That peak
OOM-kills the 31 GiB HELIOS host before CUDA starts. Here the model is created
on PyTorch's `meta` device, allocated once on CUDA, and populated tensor by
tensor from the same safetensors file. Every key and shape must match before
training; no randomly initialized parameter may survive.
"""

from __future__ import annotations

import functools
from pathlib import Path
import runpy
import sys

from safetensors import safe_open
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def install_streaming_loader() -> None:
    original_init = PI05Policy.__init__

    @functools.wraps(original_init)
    def initialize_on_meta(self, *args, **kwargs):
        config = args[0] if args else kwargs["config"]
        requested_device = config.device
        config.device = "meta"
        try:
            with torch.device("meta"):
                return original_init(self, *args, **kwargs)
        finally:
            config.device = requested_device

    @classmethod
    def stream_from_pretrained(cls, pretrained_name_or_path, *, config=None, **kwargs):
        checkpoint = Path(pretrained_name_or_path)
        weights = checkpoint / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(f"local PI0.5 safetensors checkpoint missing: {weights}")
        if config is None:
            config = PreTrainedConfig.from_pretrained(checkpoint)
        if config.device != "cuda":
            raise ValueError("streaming PI0.5 training loader requires the pinned CUDA backend")

        model = cls(config, **kwargs)
        expected_keys = set(model.state_dict())
        with safe_open(weights, framework="pt", device="cpu") as reader:
            source_keys = reader.keys()
            # The pinned HumanoidArena checkpoints already use the current
            # LeRobot names exactly. Do not invoke the upstream remapper: it
            # clones lm_head into embed_tokens before the checkpoint's own
            # embed_tokens tensor overwrites that clone, needlessly using RAM.
            if set(source_keys) != expected_keys:
                missing = sorted(expected_keys - set(source_keys))[:8]
                unexpected = sorted(set(source_keys) - expected_keys)[:8]
                raise RuntimeError(
                    f"PI0.5 checkpoint is not exact: missing={missing}, unexpected={unexpected}"
                )

            model.to_empty(device="cuda")
            destination = model.state_dict()
            loaded = set()
            with torch.no_grad():
                for source_key in source_keys:
                    tensor = reader.get_tensor(source_key)
                    target = destination[source_key]
                    if target.shape != tensor.shape:
                        raise ValueError(
                            f"PI0.5 checkpoint tensor shape mismatch: {source_key}: "
                            f"{tuple(tensor.shape)} != {tuple(target.shape)}"
                        )
                    target.copy_(tensor)
                    loaded.add(source_key)
                    del tensor
        if loaded != expected_keys:
            raise RuntimeError("PI0.5 streaming loader left an uninitialized tensor")
        model.eval()
        print(
            f"HRVLA: loaded all {len(loaded)} PI0.5 tensors to CUDA; "
            f"allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB",
            flush=True,
        )
        return model

    PI05Policy.__init__ = initialize_on_meta
    PI05Policy.from_pretrained = stream_from_pretrained


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: lerobot_pi05_low_mem_train.py LEROBOT_TRAIN.py [options]")
    upstream = Path(sys.argv[1]).resolve(strict=True)
    install_streaming_loader()
    sys.argv = [str(upstream), *sys.argv[2:]]
    print("HRVLA: exact-key meta-device PI0.5 streaming loader enabled", flush=True)
    runpy.run_path(str(upstream), run_name="__main__")


if __name__ == "__main__":
    main()
