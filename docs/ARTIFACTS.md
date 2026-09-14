# Benchmark artifact contract

HRVLA keeps source code, compact reports, plots, and logs in Git. Large model
weights and prepared LeRobot datasets live on Hugging Face and are addressed by
immutable revision plus per-file SHA-256.

## Source artifacts

- Base VLA: `nvidia/GR00T-N1.7-3B@2fc962b973bccdd5d8ce4f67cc63b264d6886495`.
- Proposal model: `nvidia/Cosmos-Reason2-2B@9ce19a195e423419c349abfc86fd07178b230561`.
- Training demonstrations:
  `nvidia/Arena-G1-Static-PickNPlace-Task@37ba80a99486a4c308477854f54b4e82e77777dc`.

The local GR00T inference subset and Cosmos snapshot were verified against these
Hub revisions with `hf cache verify`. The original Arena dataset revision is the
source of both deterministic 166/42 episode splits.

## Project artifacts

The model repository is `shin0412/HRVLA`:

- `checkpoints/GR00T-ST-RT/checkpoint-300` contains the selected subtask-trained
  checkpoint;
- `checkpoints/GR00T-STR-RT/checkpoint-300` contains the selected
  recovery-conditioned checkpoint.

The dataset repository is also named `shin0412/HRVLA`, under the dataset repo
type:

- `arena-g1-subtask-split` contains the deterministic ST training/held-out split;
- `arena-g1-recovery-split` contains the deterministic recovery-conditioned split.

Local paths are recorded in
[`benchmark-artifacts.lock.json`](../config/benchmark-artifacts.lock.json). Create
and verify the lock with:

```bash
python3 scripts/benchmark_artifacts.py create
python3 scripts/benchmark_artifacts.py verify
```

Only the predeclared final checkpoint at step 300 is publication-required.
Intermediate checkpoints, smoke runs, and failed probes remain local evidence and
must not be selected after inspecting held-out results.

Publishing changes the Hub repository revision. After upload, record the returned
model and dataset revisions in the lock, set `published` to true, regenerate the
lock without changing the artifact files, and verify all hashes again.
