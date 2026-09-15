# HumanoidArena 40-D GR00T bridge

## Purpose and claim boundary

The released HumanoidArena `sonic_refpose_v3_1` data and simulator use a
64-dimensional observation and a 40-dimensional semantic reference-pose action.
The earlier ST/STR post-training evidence uses an unrelated Arena dataset with a
43-DoF direct-joint action. Those checkpoints remain valid for their tracked
open-loop experiment, but they are not eligible for the matched HumanoidArena
closed-loop table.

The bridge in this repository is a lossless schema mapping, not a learned
controller and not a result. It enables a common HumanoidArena action-adaptation
checkpoint to be trained and shared by all five internal rows before the ST,
STR, and RT differences are applied. Closed-loop claims remain blocked until the
dataset view, checkpoint, server, simulator rollouts, scenario admission, and
evidence audit all pass.

## Frozen mapping

| Modality | Flat indices | GR00T group | Width |
| --- | ---: | --- | ---: |
| state | 0:6 | `root_heading_canonical_rot6d` | 6 |
| state | 6:35 | `joint_pos` | 29 |
| state | 35:64 | `joint_vel` | 29 |
| action | 0:2 | `root_ref_base_local_xy_delta` | 2 |
| action | 2:3 | `root_z` | 1 |
| action | 3:9 | `root_ref_rot6d` | 6 |
| action | 9:38 | `joint_pos` | 29 |
| action | 38:40 | `hand_binary` | 2 |

The action is absolute in the dataset normalization sense. Its first two values
are still the released reference-base local-XY delta defined by HumanoidArena;
the bridge must not reinterpret them as a residual from the current robot state.
Rotation 6-D uses the released row layout. The action horizon is 40 frames at
50 Hz, matching the Isaac-GR00T Unitree post-training convention.

The authoritative implementation is
[`src/hrvla_bench/humanoidarena_bridge.py`](../src/hrvla_bench/humanoidarena_bridge.py),
and the training modality configuration is
[`config/g1_humanoidarena_refpose_config.py`](../config/g1_humanoidarena_refpose_config.py).

## Leakage-resistant dataset view

HumanoidArena LeRobot v3 packs many episodes into each MP4, while the locked
Isaac-GR00T loader expects one video path per episode. Re-encoding would alter
pixels and copying each packed MP4 once per episode would waste hundreds of GB.
The view builder therefore:

1. splits episode IDs independently inside every task before training;
2. materializes only the low-dimensional Parquet rows with dense IDs;
3. creates an episode video symlink to the immutable packed source MP4;
4. records the exact starting frame as `video_frame_offset`;
5. applies that offset only at decode time through a loader shim;
6. retains the source dataset revision and split membership in a manifest.

Build the view only when the external matrix is not using the workstation's
CPU/I/O budget:

```bash
_vendor/Isaac-GR00T/.venv/bin/python \
  scripts/prepare_humanoidarena_gr00t_view.py \
  --source-root _artifacts/HumanoidArena/dataset \
  --output-root _artifacts/datasets/humanoidarena-gr00t-refpose \
  --heldout-fraction 0.2 --validation-fraction 0.1 --seed 20260915
```

The training and held-out evaluation launchers install the packed-video offset
shim automatically. Ordinary LeRobot v2 datasets have no offset field and keep
their original behavior.

## Current validation evidence

The final frozen view contains 490 training episodes (380,927 frames), 70
validation episodes (54,274 frames), and 140 hidden-test episodes (111,259
frames): exactly 70/10/20 inside each of the seven tasks. The hidden membership
is identical to the earlier frozen 80/20 view. Validation is the only split
allowed for checkpoint or hyperparameter selection; hidden-test actions and
outcomes must not be inspected until the configuration is locked.

All 700 source memberships match the manifest, all video paths are symlinks, 42
sampled low-dimensional endpoint rows reconstruct state64/action40 losslessly,
and 21 mid-episode frames spanning every task and every split match direct
packed-video decoding pixel-for-pixel. The view occupies 345,817,703 local
bytes excluding the immutable source-video targets. Its manifest SHA-256 is
`7695a09acc9d94fe3152fd4822ddae81371ed5b90360ef0825209a1132e255d3`.

See the [machine-readable validation report](../results/humanoidarena/gr00t-bridge/validation.json)
and [dataset split plot](../results/humanoidarena/gr00t-bridge/dataset_split.png).
This passes the dataset/interface gate only. It does not validate a GR00T
checkpoint or contribute an episode to the internal closed-loop table.

## Closed-loop inference server

[`scripts/serve_humanoidarena_gr00t.py`](../scripts/serve_humanoidarena_gr00t.py)
implements the same `/reset` and `/infer` wire protocol used by the released
HumanoidArena client. It splits state64 into the frozen named state groups,
routes the released task ID to its natural-language instruction, calls
`Gr00tPolicy`, and reconstructs the named predictions in the exact semantic-40
order. The server rejects a checkpoint whose embedded modality keys or horizons
drift from this contract, as well as malformed/non-finite observations or
actions. It never silently clamps, pads, truncates, or reorders a prediction.

After the shared HumanoidArena adaptation checkpoint exists, launch it from the
locked Isaac-GR00T environment:

```bash
_vendor/Isaac-GR00T/.venv/bin/python \
  scripts/serve_humanoidarena_gr00t.py \
  --model-path /path/to/shared-humanoidarena-checkpoint \
  --device cuda:0 --denoising-steps 4 --port 8000
```

The server's fake-policy HTTP integration test validates the production payload
shape without loading a model. It is interface evidence, not evidence that the
pending learned checkpoint succeeds in simulation.

## Required validation before training

- Every released task must pass the exact state/action shape and VLA protocol
  checks; a changed rotation convention or action semantics is a hard error.
- Every task must have 80 disjoint training and 20 held-out episodes under the
  frozen split seed.
- Sampled decoded frames from the view must hash identically to decoding the
  corresponding frames directly from each packed source MP4.
- A round-trip action fixture must reconstruct all 40 values without reordering.
- The shared adaptation checkpoint must be trained before the ST/STR-specific RT
  checkpoints, and its immutable revision must be used by all five rows.
- No action MSE result may be presented as task success or recovery success.
