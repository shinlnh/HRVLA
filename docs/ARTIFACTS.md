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
  recovery-conditioned checkpoint;
- `checkpoints/GR00T-ST-RT/seed-{0,1,2}/checkpoint-300` and
  `checkpoints/GR00T-STR-RT/seed-{0,1,2}/checkpoint-300` contain the six
  independently retrained replication checkpoints;
- `paper-replication/evidence` contains the six raw held-out metric files,
  aggregate summary, compact table, and SHA-256 manifest.

The dataset repository is also named `shin0412/HRVLA`, under the dataset repo
type:

- `arena-g1-subtask-split` contains the deterministic ST training/held-out split;
- `arena-g1-recovery-split` contains the deterministic recovery-conditioned split.

Published revisions:

- model repository (including the three-seed replication):
  `50ea75ab7972509c2f6827190a965484a2057bcc`;
- dataset repository: `a745b9eba09f2beac02f6f018f3c9c369b0dc5a5`.

Remote verification matched all 16 files for each of the six replication
checkpoints, all nine replication-evidence files, the original 32 checkpoint
files, and all 858 prepared dataset files. The original selected checkpoints
remain addressable at revision `5898d3177dc64322842fe1a42900cc9f952ea5e9`;
the per-group first-publication commits remain in the artifact lock for
auditability.

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

Any later replacement must publish a new Hub revision, regenerate the lock without
changing files in place, and pass both local and remote checksum verification.
