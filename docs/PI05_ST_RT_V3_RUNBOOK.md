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
The loader also regenerates all five non-persistent SigLIP/Gemma position and
rotary buffers after `to_empty`; exact checkpoint-key matching alone does not
cover these buffers. The one-step `HOI_pp_box` smoke run completed and saved a
checkpoint; an expert tensor differed from the base, proving an optimizer
update occurred. That pre-fix smoke checkpoint is **invalid for evaluation**;
it only demonstrated that the optimizer path could step.

The initial 300-step pilot at `pp_box/pilot-300-seed0-v3` is **invalid**: it
predated non-persistent-buffer restoration. The issue surfaced when its saved
checkpoint triggered a CUDA position-embedding assert during validation. Its
log and checkpoint are retained for diagnosis, never for method selection or
paper metrics. The pre-fix base-only MSE printed by that failed validator is
also invalid because its non-persistent buffers were not restored. A fresh
corrected 300-step pilot must be used instead.

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

The v2.1 view remains required by the frozen GR00T-ST-RT lock. Its Parquet
inodes have link count two and are shared with the v3 view; deleting its old
directory would reclaim only about 0.82 MiB of unshared files from the roughly
268 MiB logical view while breaking GR00T reproduction. Preserve it until that
contract is migrated and tested.

## Corrected bounded pilot (HOI_pp_box, seed zero)

The fresh run at `_artifacts/retraining/pi05-st-rt/pp_box/pilot-300-seed0-v3-corrected`
finished 300/300 optimizer steps in 313.1 s including checkpoint save. Its
loader log confirms all 813 persisted weights plus the five regenerated
non-persistent buffers. CUDA peak was 12.88 GiB allocated / 13.66 GiB
reserved. The last reported training loss was 0.048; this is *not* a
generalization metric. The run consumed 2,400 sampled frames, about 0.05 of
one epoch over this task's training frames.

The paired open-loop diagnostic used the same fixed eight frames from each of
the ten task-matched validation episodes (indices 20–29), the same diffusion
seeds, and the training-split normalization statistics for both policies. The
corrected loader was used for both base and tuned checkpoints. Macro-average
first-action MSE was 0.081130 for the released base and 0.005383 for the
300-step fine-tune; tuned was lower on all ten episodes. The full per-episode
values and checkpoint hashes live in `validation-8frames-v1.json` under the
run directory. This is a promising engineering result for one task, **not** a
HumanoidArena closed-loop success claim, a hidden-split result, or a paper
comparison across architectures.

## Next gates

1. Freeze the open-loop evaluator and run paired closed-loop HA simulator
   seeds for the corrected pilot versus its unchanged task-matched base. Do
   not select on hidden results.
2. Predeclare candidate step/seed selection and storage budget, then run the
   remaining six tasks and seeds. A one-task pilot does not make any
   PI05-ST-RT benchmark row `complete`.
3. After the GR00T-ST-RT input is migrated or archived with a tested restore
   path, re-evaluate whether to remove the 268 MiB v2.1 view locally.
