# GR00T N1.7 base-model zero-shot audit

## Scope

This audit answers only: how closely does the untouched GR00T N1.7 base model
predict actions from official recorded observations for an embodiment included
in pretraining?

It does not measure closed-loop grasp success, recovery, Unitree G1 balance, or
SONIC tracking. NVIDIA's N1.7 release lists DROID as its runnable zero-shot
benchmark example. Post-training tags and `UNITREE_G1_SONIC` cannot be tested
with the base checkpoint directly.

No training, fine-tuning, optimizer, adapter, or weight modification was run.

## Locked inputs

- Source: NVIDIA Isaac-GR00T `n1.7-release` at
  `23ace64f17aa5015259b8609d371eb61a357c776`.
- Model: `nvidia/GR00T-N1.7-3B` at
  `2fc962b973bccdd5d8ce4f67cc63b264d6886495`.
- Dataset: included `demo_data/droid_sample`.
- Embodiment: `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`.
- Runtime: Python 3.10.20, PyTorch 2.7.1+cu128, RTX 5070 Ti 15.46 GiB.
- Inference: PyTorch eager, seed 42, four denoising steps, action horizon 8.

## Exact object-placement run

```bash
uv run python scripts/deployment/standalone_inference_script.py \
  --model-path /absolute/path/to/GR00T-N1.7-3B \
  --dataset-path demo_data/droid_sample \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --traj-ids 1 \
  --steps 266 \
  --inference-mode pytorch \
  --denoising-steps 4 \
  --action-horizon 8 \
  --seed 42
```

Task: `Put the blue block in the green bowl`.

- 266 recorded steps, 17 action dimensions.
- Unnormalized MSE: `0.0038425252`.
- Unnormalized RMSE: `0.0619881055`.
- Unnormalized MAE: `0.0383970094`.
- Mean inference time after warm-up: `105.5 ms` (~9.48 Hz).
- P90 inference time: `108.2 ms`.
- Model load time on the repeated run: `17.91 s`.

The trajectory plot shows that most action dimensions follow the demonstrated
trend, but the gripper dimension has visible timing and magnitude error. Because
grasp success is sensitive to that event, a small aggregate action MSE is not
evidence that the object would be grasped in closed loop.

## Official two-trajectory smoke test

For the first 200 steps of the two non-empty DROID examples:

| Task | MSE | MAE |
|---|---:|---:|
| Put the blue block in the green bowl | 0.003282 | 0.037574 |
| Move the bottom-right duvet tip left | 0.036891 | 0.116728 |
| Mean over the two trajectories | 0.020087 | 0.077151 |

The second task has 11.24 times the MSE and 3.11 times the MAE of the object
placement task. The honest conclusion is that zero-shot action prediction is
promising on the object-placement recording but uneven across tasks. A simulator
rollout is required before reporting task success rate.

Machine-readable values are in
[`reports/zero_shot_n1_7_droid.json`](../reports/zero_shot_n1_7_droid.json).
