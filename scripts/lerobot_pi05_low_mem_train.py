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
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

from safetensors import safe_open
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import factory as policy_factory
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


TOKENIZER_SHA256 = "ef6773c135b77b834de1d13c75a4c98ab7a3684ffd602d1831e1f1bf5467c563"
TOKENIZER_CONFIG_SHA256 = "3259402b1d1802e02417d7bff75a889ec61d359d15be6050a957b307c48edbbe"
STALE_TOKENIZER = "/ai/Yichi/taowen/ckpts/checkpoints/paligemma-3b-pt-224"


def install_tokenizer_override() -> None:
    tokenizer_dir = Path(os.environ["HRVLA_PI05_TOKENIZER_DIR"]).resolve(strict=True)
    tokenizer_file = tokenizer_dir / "tokenizer.json"
    digest = hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()
    if digest != TOKENIZER_SHA256:
        raise ValueError("local PaliGemma tokenizer does not match the pinned artifact")
    config_digest = hashlib.sha256((tokenizer_dir / "tokenizer_config.json").read_bytes()).hexdigest()
    if config_digest != TOKENIZER_CONFIG_SHA256:
        raise ValueError("local PaliGemma tokenizer config does not match the pinned artifact")
    original = policy_factory.make_pre_post_processors

    @functools.wraps(original)
    def use_local_tokenizer(*args, **kwargs):
        checkpoint = kwargs.get("pretrained_path")
        if checkpoint:
            payload = json.loads(
                (Path(checkpoint) / "policy_preprocessor.json").read_text(encoding="utf-8")
            )
            steps = [step for step in payload.get("steps", [])
                     if step.get("registry_name") == "tokenizer_processor"]
            if len(steps) != 1 or steps[0].get("config", {}).get("tokenizer_name") != STALE_TOKENIZER:
                raise ValueError("checkpoint tokenizer contract differs from pinned PaliGemma artifact")
            overrides = kwargs.setdefault("preprocessor_overrides", {})
            tokenizer_override = overrides.setdefault("tokenizer_processor", {})
            tokenizer_override["tokenizer_name"] = str(tokenizer_dir)
        return original(*args, **kwargs)

    policy_factory.make_pre_post_processors = use_local_tokenizer
    print(f"HRVLA: pinned local PaliGemma tokenizer {digest[:12]}", flush=True)


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
    install_tokenizer_override()
    sys.argv = [str(upstream), *sys.argv[2:]]
    print("HRVLA: exact-key meta-device PI0.5 streaming loader enabled", flush=True)
    runpy.run_path(str(upstream), run_name="__main__")


if __name__ == "__main__":
    main()
