# HumanoidArena admission

HumanoidArena is an external benchmark input, not part of `ours`. Its source,
released models, dataset, runtime contract, task launchers, and admission rules
are frozen in `config/humanoidarena-admission.lock.json`.

The official release requires Isaac Sim 5.0.0 and Isaac Lab
`release/2.2.0`. HRVLA's existing workstation runtime is Isaac Sim 5.1.0 and
Isaac Lab 2.3.2, so HumanoidArena uses an isolated Python 3.11 environment at
Isaac Lab revision `46dff135f44683f031edf346e544fcfd8456b2bb`. SONIC uses
ONNX Runtime GPU 1.22.0 in that environment.

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

The released PI0.5 checkpoints refer to Google's gated
`google/paligemma-3b-pt-224` tokenizer, which was unavailable to the executing
Hub account. Runtime evidence therefore uses only the tokenizer files from
public repository `leo009/paligemma-3b-pt-224` at immutable revision
`39996beb6fb17c5d16a50d3ef8f7a96ad9d03986`. Their SHA-256 values are locked in
the admission file. This compatibility substitution is disclosed and must not
be described as an official Google snapshot.

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

## Current preflight

The 2026-09-15 local preflight passes 39 of 41 checks. The exact locked source,
seven task launchers, four evaluation modes, extracted `objects/` and `robots/`
assets, SONIC policy files, and the selected official model and dataset inputs
are present. All seven base-test scenes reset and completed 64 CUDA physics
steps in the isolated Isaac Sim 5.0.0 runtime. A bounded Boxing integration run
also completed 40 steps through camera observation, CPU PI0.5 inference,
CUDA SONIC encoder/decoder execution, action application, recording, and result
serialization. PI0.5 is deliberately kept on CPU with a direct safetensors
loader while simulation and SONIC occupy the 16 GB GPU; this prevents the
upstream loader's transient float32 allocation from exhausting system memory.
The same path subsequently completed a full-horizon Boxing seed-0 episode with
success at step 199 of a 900-step limit (12 VLA requests, 94.918 seconds). This
single episode establishes closed-loop executability, not an aggregate
benchmark success rate or oracle admission.

Two gates remain blocked:

1. The latest upstream `release/open-source-prep` revision is still
   `68479287a784a69be9ce6ad739311d2f11f75ef9`, and its own tests still report
   8 passed and 2 failed for the OpenDoor articulation/USD mismatch described
   above.
2. No independent 20/20 oracle admission file exists yet. It cannot be created
   honestly until the upstream runtime contract is consistent and the oracle
   trials have actually run.

The machine-readable preflight is retained at
[`results/humanoidarena/admission/preflight.json`](../results/humanoidarena/admission/preflight.json),
and the unmodified upstream test result is retained as
[`upstream-self-test.xml`](../results/humanoidarena/admission/upstream-self-test.xml).
The seven task smokes and bounded end-to-end evidence are indexed by
[`results/humanoidarena/runtime/summary.json`](../results/humanoidarena/runtime/summary.json).
