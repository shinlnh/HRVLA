# HumanoidArena admission

HumanoidArena is an external benchmark input, not part of `ours`. Its source,
released models, dataset, runtime contract, task launchers, and admission rules
are frozen in `config/humanoidarena-admission.lock.json`.

The official release requires Isaac Sim 5.0.0 and Isaac Lab
`release/2.2.0`. HRVLA's existing workstation runtime is Isaac Sim 5.1.0 and
Isaac Lab 2.3.2, so HumanoidArena must use a separate environment. Do not
silently run it in the existing runtime and label the result official.

At the locked source revision, the two lightweight upstream runtime test files
report 8 passed and 2 failed. Both failures are OpenDoor contract mismatches
between the tests and implementation. The preflight preserves this as a hard
blocker; update the source revision only when upstream publishes a consistent
release. Patching the ignored vendor checkout is not accepted as official
HumanoidArena evidence.

The released Hugging Face resources are intentionally not downloaded by the
preflight command: the model repository is about 572 GB at the locked revision,
and only the checkpoint families selected for an experiment should be fetched.
The dataset and model locations are:

- dataset: `hf://datasets/WilliamWang16/HumanoidArena_dataset_v3_1`
- models: `hf://models/WilliamWang16/HumanoidArena_models`
- HRVLA ST/STR retraining artifacts: `hf://models/shin0412/HRVLA` and
  `hf://datasets/shin0412/HRVLA`

Run the read-only preflight with explicit runtime versions:

```bash
PYTHONPATH=src python3 -m hrvla_bench.cli check-humanoidarena \
  config/humanoidarena-admission.lock.json \
  --isaac-sim-version 5.0.0 \
  --isaac-lab-ref release/2.2.0 \
  --output outputs/humanoidarena-admission.json
```

The command exits with status 3 while any prerequisite is absent. Admission
requires the exact source revision, all seven official task launchers, all four
evaluation modes, simulation assets, SONIC policy files, released checkpoints,
the dataset, the isolated runtime, and independent `20/20` oracle evidence.
Merely cloning the source or seeing a checkpoint on the Hub is not sufficient
for a claim-bearing evaluation.
