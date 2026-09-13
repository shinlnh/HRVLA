# HRVLA baseline reproduction

This branch is the reproducible baseline substrate for a Unitree G1 humanoid
driven by NVIDIA GR00T N1.7 and the official GEAR-SONIC whole-body controller.
It does not contain a replacement controller or a locally invented robot
interface.

The upstream sources are locked in [`config/upstreams.lock.json`](config/upstreams.lock.json):

- NVIDIA Isaac-GR00T N1.7 for the VLA policy server;
- NVIDIA GR00T-WholeBodyControl / GEAR-SONIC for the 64-dimensional latent
  action interface and 50 Hz whole-body control;
- Unitree `unitree_sim_isaaclab` for the G1 simulation/DDS reference;
- NVIDIA Isaac Lab 2.3.2 on Isaac Sim 5.1 for the simulation runtime.

Two evaluation profiles are intentionally distinct:

- zero-shot base-model audit: the N1.7 release source, a baked-in pretrain
  embodiment, and no fine-tuning;
- Unitree G1 + SONIC integration: current locked integration source and the
  `UNITREE_G1_SONIC` post-training embodiment.

The base checkpoint is not a zero-shot SONIC checkpoint. See
[`docs/ZERO_SHOT_AUDIT.md`](docs/ZERO_SHOT_AUDIT.md) for the first measured
zero-shot run and its limitations.

The VLA and simulator/controller dependencies require different Python
versions. Keep them in separate environments:

- `hrvla-sim`: Python 3.11, Isaac Sim 5.1, Isaac Lab 2.3.2, GEAR-SONIC;
- `hrvla-vla`: Python 3.12, Isaac-GR00T N1.7.

## Start here

```bash
python3 scripts/verify_lock.py
python3 scripts/bootstrap_upstreams.py
```

The bootstrap command clones only source code and checks out immutable commits.
It deliberately does not install packages or download model/data artifacts.
Follow [`docs/BASELINE.md`](docs/BASELINE.md) for the official install, artifact,
preflight, and smoke-test commands.

The tested SONIC-only simulator baseline is documented in
[`docs/SONIC_BASELINE.md`](docs/SONIC_BASELINE.md). After its isolated runtime
and locked artifacts are present, run either:

```bash
python3 scripts/run_sonic_release.py metrics
python3 scripts/run_sonic_release.py viewer
```

## Verified Isaac Sim demo

The official SONIC checkpoint was validated on an RTX 5070 Ti with Isaac Sim
5.1.0, Isaac Lab 2.3.2, and NVIDIA driver 580.173.02. The G1 completed both
bundled walk-forward motions (`4,004` frames) without termination. See the
[machine-readable result](results/sonic-default-sample.json) and the full
[reproduction notes](docs/SONIC_BASELINE.md).

![GEAR-SONIC controlling Unitree G1 in Isaac Sim 5.1](docs/assets/sonic-isaac-sim-5.1.png)

## Long-horizon subtask planning

This branch adds an independent implementation of a selective, world-model-guided
test-time-compute planner. It decomposes a large instruction into one executable
language subtask at a time, verifies it against observed affordances, and sends that
bounded instruction through GR00T's existing language input. The 64-dimensional
GR00T/SONIC action contract is not changed.

Validate the benchmark and run all five algorithm variants:

```bash
python3 scripts/subtask_planner.py validate
python3 scripts/subtask_planner.py simulate \
  --output-dir results/subtask/cpu-noise-028 \
  --episodes-per-task 1000 --workers 0 --proposal-error-rate 0.28
```

Evaluate real NVIDIA Cosmos-Reason2-2B proposals on GPU with a local snapshot:

```bash
_vendor/Isaac-GR00T/.venv/bin/python scripts/subtask_planner.py gpu-eval \
  --model-path "$COSMOS_REASON2_SNAPSHOT" \
  --output-dir results/subtask/cosmos-reason2
```

See [`docs/SUBTASK_PLANNER.md`](docs/SUBTASK_PLANNER.md) for the algorithm,
paper comparison, metric definitions, integration contract, and limitations.
Measured CPU/GPU/SONIC results and negative findings are in
[`docs/SUBTASK_RESULTS.md`](docs/SUBTASK_RESULTS.md).

## Transition-aware subtask recovery

The recovery coordinator catches a declared subtask failure, gates recovery on
observed readiness, and selects a corrective primitive whose exit satisfies the
interrupted task's handoff contract. Its BATON-inspired transition memory operates
above the same GR00T language adapter and leaves SONIC's action interface unchanged.

Validate the protocol and run the paired-seed recovery benchmark:

```bash
python3 scripts/subtask_recovery.py validate
python3 scripts/subtask_recovery.py simulate \
  --output-dir results/recovery/cpu-fault-015 \
  --episodes-per-task 3000 --workers 32 --disturbance-rate 0.15
```

Evaluate 20 independent candidate sets from the locked Cosmos-Reason2-2B snapshot:

```bash
_vendor/Isaac-GR00T/.venv/bin/python scripts/subtask_recovery.py gpu-eval \
  --model-path "$COSMOS_REASON2_SNAPSHOT" \
  --output-dir results/recovery/cosmos-reason2-r20 \
  --candidates-per-failure 3 --repetitions 20
```

See [`docs/SUBTASK_RECOVERY.md`](docs/SUBTASK_RECOVERY.md) for the algorithm,
paper comparison, metric definitions, and safety boundary. The measured report,
retained raw traces, charts, Cosmos outputs, SONIC metrics, video, and Isaac Sim
frames are indexed by
[`docs/SUBTASK_RECOVERY_RESULTS.md`](docs/SUBTASK_RECOVERY_RESULTS.md).

Run the low-level disturbance regression and record its Isaac Sim camera output:

```bash
python3 scripts/run_sonic_release.py metrics --inject-push --num-envs 32 \
  --output-dir results/recovery/isaacsim/sonic-push-32env
python3 scripts/run_sonic_release.py record --inject-push \
  --metrics-file results/recovery/isaacsim/sonic-push-32env/metrics_eval.json \
  --output-dir results/recovery/isaacsim/push-recording
```

## In-domain VLA post-training

The subtask-conditioned GR00T N1.7 action decoder was post-trained on a
leakage-resistant 166/42 episode split of NVIDIA's Isaac Lab G1 pick-and-place
data. Paired evaluation across all held-out episodes and five observation
conditions reduces clean action MSE by 4.988%; per-phase results also retain the
approach/grasp regressions. The experiment includes immutable inputs, raw
hardware telemetry, checkpoint trends, statistical tests, Isaac Lab evidence,
and a fresh recorded Isaac Sim/SONIC regression.

See [`docs/SUBTASK_VLA_RETRAINING.md`](docs/SUBTASK_VLA_RETRAINING.md) for the
exact training scope, commands, metrics, plots, videos, and claim boundaries.
