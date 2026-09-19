#!/usr/bin/env python3
"""Memory-bounded GR00T N1.7 VLA post-training for a 16 GiB workstation.

The official default tunes more than 1.6B action-head parameters and documents
a 40 GiB minimum.  This launcher keeps the Cosmos vision/language backbone and
the flow-matching core frozen, then trains the embodiment-conditioned final
action decoder.  The resulting checkpoint is a real GR00T VLA checkpoint, but
the claim is deliberately limited to action-decoder adaptation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import torch


def _load_modality_config(path: Path) -> None:
    spec = importlib.util.spec_from_file_location("hrvla_retrain_modality", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import modality config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _install_action_decoder_profile(save_dir: Path) -> None:
    """Patch the upstream pipeline at its model-construction seam."""
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline

    original_create_model = Gr00tN1d7Pipeline._create_model

    def create_action_decoder_model(pipeline):
        model = original_create_model(pipeline)
        model.requires_grad_(False)
        model.to(dtype=torch.bfloat16)
        model.action_head.action_decoder.to(dtype=torch.float32)
        model.action_head.action_decoder.requires_grad_(True)

        # Upstream's train-mode hook keys off this runtime flag.  The other
        # projector modules remain frozen through requires_grad=False.
        model.action_head.tune_projector = True
        trainable = [
            {"name": name, "parameters": parameter.numel(), "dtype": str(parameter.dtype)}
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        manifest = {
            "schema_version": 1,
            "profile": "action_decoder_only",
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(row["parameters"] for row in trainable),
            "trainable_tensors": trainable,
            "frozen_backbone": True,
            "frozen_flow_matching_core": True,
            "parameter_dtype_policy": "frozen_bf16_trainable_decoder_fp32",
        }
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "trainable_parameters.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    key: manifest[key]
                    for key in ("profile", "total_parameters", "trainable_parameters")
                }
            )
        )
        return model

    Gr00tN1d7Pipeline._create_model = create_action_decoder_model


def _install_redundant_final_save_guard(max_steps: int, save_steps: int) -> None:
    """Skip the upstream root save when the final numbered checkpoint exists.

    Hugging Face already writes ``checkpoint-<max_steps>`` when the final step
    is aligned with ``save_steps``.  GR00T then calls ``trainer.save_model()``
    once more at the output root, serializing the same 3.1B-parameter model a
    second time.  The numbered checkpoint is the locked evaluation artifact,
    so the duplicate adds I/O latency and disk use without adding recoverability.
    """
    if max_steps % save_steps:
        return

    from gr00t.experiment.trainer import Gr00tTrainer

    original_save_model = Gr00tTrainer.save_model

    def save_model_without_duplicate(trainer, output_dir=None, *args, **kwargs):
        final_checkpoint = (
            Path(trainer.args.output_dir) / f"checkpoint-{trainer.state.global_step}"
        )
        final_state = final_checkpoint / "trainer_state.json"
        is_final_root_save = (
            output_dir is None
            and trainer.state.global_step == max_steps
            and final_state.is_file()
        )
        if is_final_root_save:
            print(
                json.dumps(
                    {
                        "checkpoint": str(final_checkpoint),
                        "event": "skip_redundant_final_root_save",
                        "global_step": trainer.state.global_step,
                    },
                    sort_keys=True,
                )
            )
            return None
        return original_save_model(trainer, output_dir, *args, **kwargs)

    Gr00tTrainer.save_model = save_model_without_duplicate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--modality-config-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--dataloader-num-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state-dropout-prob", type=float, default=0.2)
    parser.add_argument("--color-jitter", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GR00T VLA post-training")
    if args.max_steps < 1 or args.save_steps < 1:
        raise ValueError("max_steps and save_steps must be positive")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient accumulation must be positive")
    if args.global_batch_size < 1:
        raise ValueError("global batch size must be positive")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    _load_modality_config(args.modality_config_path.resolve())

    # HumanoidArena LeRobot-v3 views keep immutable packed videos as symlinks
    # and record each episode's exact starting frame in episodes.jsonl.  The
    # patch is a no-op for ordinary v2 datasets (offset defaults to zero).
    from humanoidarena_gr00t_video import install_packed_video_offset_patch

    install_packed_video_offset_patch()

    from gr00t.configs.base_config import get_default_config
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.experiment import run

    embodiment = EmbodimentTag.NEW_EMBODIMENT.value
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [str(args.dataset_path.resolve())],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment,
                    }
                ],
            }
        }
    )
    config.load_config_path = None
    config.data.seed = args.seed
    config.data.shard_size = 512
    config.data.episode_sampling_rate = 0.1
    config.data.num_shards_per_epoch = 2_000

    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = False
    config.model.tune_diffusion_model = False
    config.model.tune_vlln = False
    config.model.load_bf16 = True
    config.model.backbone_trainable_params_fp32 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.use_relative_action = True
    config.model.state_dropout_prob = args.state_dropout_prob
    config.model.color_jitter_params = (
        {"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}
        if args.color_jitter
        else None
    )

    config.training.start_from_checkpoint = str(args.base_model_path.resolve())
    config.training.transformers_local_files_only = True
    config.training.output_dir = str(args.output_dir.resolve())
    config.training.experiment_name = None
    config.training.num_gpus = 1
    config.training.global_batch_size = args.global_batch_size
    config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.training.dataloader_num_workers = args.dataloader_num_workers
    config.training.learning_rate = args.learning_rate
    config.training.max_steps = args.max_steps
    config.training.save_steps = args.save_steps
    config.training.save_total_limit = 3
    config.training.save_only_model = True
    config.training.logging_steps = 1
    config.training.optim = "adamw_torch_fused"
    config.training.bf16 = True
    config.training.fp16 = False
    config.training.tf32 = True
    config.training.eval_strategy = "no"
    config.training.use_wandb = False

    _install_action_decoder_profile(args.output_dir.resolve() / "experiment_cfg")
    _install_redundant_final_save_guard(args.max_steps, args.save_steps)
    run(config)


if __name__ == "__main__":
    main()
