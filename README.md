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
