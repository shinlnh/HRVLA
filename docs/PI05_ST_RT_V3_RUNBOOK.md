# PI0.5 ST-RT v3 data and training runbook

Updated 2026-09-22. This records engineering readiness, **not** a v2 paper
benchmark result or an in-domain checkpoint selection.

## Data provenance and validation

The pinned HumanoidArena SONIC source is already LeRobot v3.0. The project's
weakly labelled ST training view is LeRobot v2.1 because its labels were
materialized as per-episode Parquet files; each episode's MP4 symlink points to
an entire packed v3 source video, with a separate frame offset. The stock
v2.1-to-v3 converter must not be run on this view.

`scripts/export_humanoidarena_st_v3.py` audits all source episode lengths,
global/frame/task indices, source-v3 video paths and frame offsets, then makes
hard links to unchanged Parquet/MP4 bytes and writes v3 metadata. It is
non-destructive and refuses to overwrite an output. The new roots are:

- `_artifacts/datasets/humanoidarena-st-rt-v3/train`: 490 episodes,
  380,927 frames, 26 packed-video shards, 16 sub-task prompts.
- `_artifacts/datasets/humanoidarena-st-rt-v3/validation`: 70 episodes,
  54,274 frames, 23 packed-video shards.

`scripts/verify_humanoidarena_st_v3.py` checked all 560 episode-Parquet and 49
split-specific video hard links and their metadata, then loaded 14 train and seven validation probes
through LeRobot, comparing state/action/task labels with the source and
decoding images. Both `meta/hrvla_export_audit.json` files say `loader_pass`.
The nonnumeric legacy `__fingerprints__` entry was removed from v3 `stats.json`
because LeRobot's normalizer tensorizes every stats key; the original v2.1
files were left untouched. The source dataset and label audit SHA-256 values
are retained in the v3 info/audit files.

Do **not** delete the v2.1 ST view yet: `config/humanoidarena-rt-training.lock.json`
still names it as the GR00T-ST-RT input. The v3 PI0.5 view is independent
because its Parquet bytes are hard-linked, but deleting v2.1 before migrating
the GR00T path would break historical reproduction for negligible disk gain
(the v2.1 directories are only about 268 MiB).

## PI0.5 adaptation contract

The released PI0.5 HA baseline has one checkpoint per task. Fine-tune each
task from its **matching** released checkpoint on only that task's 70 train
episodes; use the corresponding ten validation episodes for checkpoint/seed
selection. Do not mix all seven tasks into one task-specific checkpoint. The
training launcher enforces a source-dataset filter, matching checkpoint path,
the `loader_pass` audit and train-only split. Heldout stays unopened.

The pinned model has 4,143,421,208 parameters; expert-only adaptation updates
693,438,504 of them. A project-owned loader constructs the model on PyTorch
`meta`, then streams all 813 exactly matching checkpoint tensors into CUDA.
This preserves the released weights and avoids LeRobot's host-RAM OOM during
ordinary full-model construction. It also binds the released preprocessor's
stale absolute tokenizer path to the pinned local PaliGemma tokenizer by SHA.
The one-step `HOI_pp_box` smoke run completed and saved a checkpoint; an expert
tensor differed from the base, proving an optimizer update occurred. This
smoke checkpoint is **not** a selected method checkpoint.

## Memory-safe throughput probe (HOI_pp_box, 10 steps, no checkpoint)

All four probes used BF16, gradient checkpointing, expert-only training and
four data-loader workers. `data_s` was about 0.001–0.005 s after warmup, so
CPU input is not the steady-state bottleneck. Approximate steady-state figures
from LeRobot's per-step output:

| Physical batch | Update seconds | Samples/second | CUDA peak allocated | Outcome |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.182 | 5.5 | 12.85 GiB | pass |
| 2 | 0.296 | 6.8 | 12.86 GiB | pass |
| 4 | 0.510 | 7.8 | 12.86 GiB | pass |
| 8 | 0.942 | 8.5 | 12.88 GiB | pass; reserved 13.66 GiB |

Batch eight gives the highest observed sample throughput with a 2.6 GiB
reserved-VRAM margin on the 16 GiB RTX 5070 Ti. More workers will not
materially help this profile and previously exhausted host RAM at 28 workers.
These are infrastructure measurements, not paper metrics.

## Next gates

1. Run a bounded task-matched 300-step pilot on `HOI_pp_box`, seed zero, batch
   eight/workers four; retain its exact command, source/checkpoint hashes,
   optimizer log and final checkpoint.
2. Evaluate that checkpoint and its unchanged base on the same ten validation
   episodes with one fixed open-loop metric and, later, paired closed-loop HA
   simulator seeds. Do not select on hidden results.
3. Predeclare candidate step/seed selection and storage budget, then run the
   remaining six tasks and seeds. A completed train alone does not make the
   PI05-ST-RT benchmark rows `complete`.
4. After the GR00T-ST-RT input is migrated or archived with a tested restore
   path, re-evaluate whether to remove the 268 MiB v2.1 view locally.
