# Official baseline contract

## What is reproduced

The baseline follows NVIDIA's documented Unitree G1 VLA path:

```text
camera + proprioception -> GR00T N1.7 policy server
                       -> UNITREE_G1_SONIC latent actions
                       -> GEAR-SONIC whole-body controller at 50 Hz
                       -> Unitree G1 / Isaac Lab simulation
```

`UNITREE_G1_SONIC` is a post-training embodiment tag. The base
`nvidia/GR00T-N1.7-3B` checkpoint must be fine-tuned for it; the lock file does
not imply that the base checkpoint is directly inference-ready for SONIC.

The compatibility tuple used here is the intersection of official requirements:

- GEAR-SONIC training: Ubuntu 22.04+, CUDA 12.x, Python 3.11, Isaac Lab 2.3+;
- Isaac Lab 2.3.x: built on Isaac Sim 5.1;
- Unitree simulator: supports Isaac Sim 5.1 with Python 3.11;
- Isaac-GR00T N1.7 release: Python `>=3.12,<3.13`.

That is why the policy server and simulator/controller are isolated processes.
Do not `pip install` both projects into one environment.

## 1. Verify and fetch source code

```bash
python3 scripts/verify_lock.py
python3 scripts/bootstrap_upstreams.py
python3 scripts/verify_lock.py --check-local
```

All repositories are left in detached-HEAD state under `_vendor/`. The exact
commits and Hugging Face snapshot revisions are in
`config/upstreams.lock.json`.

## 2. Simulator/controller environment (`hrvla-sim`)

Create a Python 3.11 environment, install Isaac Sim 5.1 and the locked Isaac
Lab checkout using the official Isaac Lab installation instructions. Then run
the GEAR-SONIC commands from its locked checkout. The commands below are copied
from the locked NVIDIA training guide:

```bash
cd _vendor/GR00T-WholeBodyControl
pip install -e "gear_sonic/[training]"
pip install huggingface_hub
python download_from_hf.py --training
python check_environment.py --training
python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  num_envs=16 headless=True \
  ++algo.config.num_learning_iterations=5
```

The training artifact download is approximately 30 GB. It is intentionally not
part of the source bootstrap.

For the official Unitree simulation/DDS reference, use the locked
`_vendor/unitree_sim_isaaclab` checkout. Its `auto_setup_env.sh 5.1 ...` script
performs host package installation, creates a Conda environment, builds DDS,
downloads assets, and writes certificates. Review it before running; HRVLA does
not execute those host mutations automatically.

## 3. VLA environment (`hrvla-vla`)

Create a separate Python 3.12 environment and install the locked Isaac-GR00T
release according to its README. NVIDIA's documented SONIC fine-tuning command
uses:

```bash
uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /absolute/path/to/lerobot-dataset \
  --embodiment-tag UNITREE_G1_SONIC \
  --modality-config-path gr00t/configs/data/embodiment_configs.py
```

For inference, NVIDIA documents:

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path /absolute/path/to/finetuned-checkpoint \
  --embodiment-tag UNITREE_G1_SONIC \
  --device cuda:0
```

Dataset paths and checkpoint paths are placeholders by design; they must point
to artifacts produced or selected for a concrete experiment.

## 4. Artifact ownership

Official baseline inputs remain in the NVIDIA Hugging Face repositories locked
in the manifest. Project-produced large files and benchmark outputs belong in
`shin0412/HRVLA`; benchmark datasets belong in the dataset repository
`shin0412/HRVLA`. Neither project repository is treated as an upstream baseline
source.

## Primary sources

- [GEAR-SONIC training installation](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/087f9ac01d46f6d8e4d0b73c01ae64799f292a38/docs/source/getting_started/installation_training.md)
- [GEAR-SONIC VLA workflow](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/087f9ac01d46f6d8e4d0b73c01ae64799f292a38/docs/source/tutorials/vla_workflow.md)
- [Isaac-GR00T N1.7 release](https://github.com/NVIDIA/Isaac-GR00T/tree/23ace64f17aa5015259b8609d371eb61a357c776)
- [Unitree Isaac Lab simulator](https://github.com/unitreerobotics/unitree_sim_isaaclab/tree/e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc)
- [Isaac Lab 2.3.2](https://github.com/isaac-sim/IsaacLab/tree/37ddf626871758333d6ed89cf64ad702aef127d0)
