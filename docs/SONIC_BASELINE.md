# SONIC release baseline

This branch uses the original NVIDIA GEAR-SONIC release as the fixed whole-body
execution baseline for Unitree G1. The source revision, Hugging Face snapshot,
artifact hashes, runtime versions, and controller contract are recorded in
[`config/sonic-release.lock.json`](../config/sonic-release.lock.json).

## What this run proves

The release smoke test loads the official 64-dimensional SONIC latent model,
decodes it to 29 G1 joint-position actions, and executes the controller at
50 Hz in Isaac Sim 5.1 / Isaac Lab 2.3.2. The included two walk-forward motions
contain 4,004 frames (80.04 seconds total).

This is a controller motion-tracking test. It is not an apple-grasp task, a
DROID replay, a VLA task-success result, or a recovery benchmark. A manipulation
task requires a GR00T checkpoint post-trained for `UNITREE_G1_SONIC` plus a
scene/dataset with matching observations, hands, actions, and success criteria.
The base `nvidia/GR00T-N1.7-3B` checkpoint cannot be claimed as a zero-shot G1
manipulation policy.

## Run the locked release

The local simulator environment lives under `_vendor/sonic-runtime`; large
upstream sources and checkpoints stay ignored by Git. The runner verifies the
source commits and SHA-256 hashes before starting Isaac Sim.

Headless metrics for the two official sample motions:

```bash
python3 scripts/run_sonic_release.py metrics
```

The result is written to
`_artifacts/sonic_eval/default_sample/metrics_eval.json`.

The first validated run on this machine completed both bundled motions without
termination: success rate `1.0`, progress rate `1.0`, local MPJPE `23.55 mm`,
and Procrustes-aligned MPJPE `17.31 mm`. The compact, reviewable record is in
[`results/sonic-default-sample.json`](../results/sonic-default-sample.json).
These two walk-forward samples are a smoke result only and are not evidence of
general task performance.

Open the Isaac Sim viewer:

```bash
python3 scripts/run_sonic_release.py viewer
```

The viewer follows NVIDIA's official quick-start behavior and continues until
`Ctrl+C`. The runner removes inherited `ISAAC_PATH`, `ISAACLAB_PATH`,
`PYTHONPATH`, and `LD_LIBRARY_PATH` so the host's Isaac Sim 6.0 installation
cannot silently replace the locked 5.1 runtime.

## Evaluation contract

Use SONIC as the shared execution backend, not as the judge. In the recovery
benchmark, all methods being compared must use the same SONIC variant,
checkpoint, physics settings, control frequency, task initial states, event
boundaries, and failure seeds. Report task success and recovery metrics from
the benchmark harness separately from SONIC motion-tracking metrics.

Do not compare the original, low-latency, and v1.1 checkpoints in one table
without naming the variant. They differ in future-reference representation and
intended deployment. This branch locks the original default release: 10 future
frames at 20 ms spacing, approximately 200 ms of reference lookahead.
