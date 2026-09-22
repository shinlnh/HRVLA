#!/usr/bin/env python3
"""Serve a HumanoidArena-compatible semantic-40 action chunk from GR00T."""

from __future__ import annotations

import argparse
import importlib.util
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrvla_bench.humanoidarena_gr00t_http import (  # noqa: E402
    HumanoidArenaGr00tState,
    make_handler,
    validate_policy_contract,
)
from hrvla_subtask.http_routing import ObservedStateRouter  # noqa: E402


def _load_modality_config(path: Path) -> None:
    spec = importlib.util.spec_from_file_location("hrvla_ha_inference_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import modality config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--modality-config-path",
        type=Path,
        default=ROOT / "config/g1_humanoidarena_refpose_config.py",
    )
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-instruction")
    parser.add_argument("--subtask-program", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.denoising_steps < 1:
        raise ValueError("denoising steps must be positive")
    if not 0 <= args.port <= 65535:
        raise ValueError("port must be between 0 and 65535")

    _load_modality_config(args.modality_config_path.resolve())
    import torch
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=str(args.model_path.resolve()),
        device=args.device,
    )
    policy.model.action_head.num_inference_timesteps = args.denoising_steps
    validate_policy_contract(policy)
    router = (
        ObservedStateRouter.from_path(args.subtask_program)
        if args.subtask_program is not None else None
    )
    state = HumanoidArenaGr00tState(
        policy, default_instruction=args.default_instruction, instruction_router=router
    )
    state.reset()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(
        json.dumps(
            {
                "event": "ready",
                "host": args.host,
                "port": server.server_address[1],
                "model_path": str(args.model_path.resolve()),
                "embodiment_tag": args.embodiment_tag,
                "action_contract": "semantic_v3_state64_action40",
                "denoising_steps": args.denoising_steps,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
