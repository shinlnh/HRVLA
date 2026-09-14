# Three-seed VLA retraining replication

The locked ST-RT and STR-RT action-decoder experiments were independently
retrained with seeds 0, 1, and 2. Every run reached the predeclared step-300
checkpoint. Each final checkpoint was then evaluated on all 42 held-out
trajectories under clean, vision-noise, occlusion, state-noise, and combined
conditions. Evaluation used 120 frames per trajectory, execution horizon 40,
four denoising steps, and evaluation seed 20260913.

## Multi-seed result

Values are held-out action MSE, reported as mean ± sample standard deviation
over the three training seeds.

| Condition | ST-RT | STR-RT | STR-RT reduction | Hierarchical bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0.100809 ± 0.000120 | **0.100233 ± 0.000024** | 0.571% | [0.000429, 0.000721] |
| Vision noise | 0.100809 ± 0.000122 | **0.100349 ± 0.000020** | 0.455% | [0.000311, 0.000610] |
| Occlusion | 0.100793 ± 0.000123 | **0.100036 ± 0.000037** | 0.752% | [0.000603, 0.000910] |
| State noise | 0.100808 ± 0.000120 | **0.100231 ± 0.000025** | 0.573% | [0.000430, 0.000723] |
| Combined | 0.100776 ± 0.000124 | **0.100207 ± 0.000025** | 0.565% | [0.000414, 0.000722] |

STR-RT improves over the seed-matched ST-RT checkpoint for all three training
seeds in every condition. The confidence interval uses 20,000 hierarchical
bootstrap resamples: training seeds are resampled first, then trajectories
within each selected seed. This preserves the experiment's true replication
unit instead of treating all 126 seed-trajectory pairs as independent seeds.

The machine-readable aggregate is
[`results/retraining/paper-seeds/summary.json`](../results/retraining/paper-seeds/summary.json),
the compact table is
[`condition_summary.csv`](../results/retraining/paper-seeds/condition_summary.csv),
and the six raw evaluation files plus their SHA-256 values are retained under
[`results/retraining/paper-seeds/raw/`](../results/retraining/paper-seeds/raw/).

## Reproduction

```bash
PYTHONPATH=src python3 scripts/run_paper_retraining.py
PYTHONPATH=src python3 scripts/summarize_paper_retraining.py
PYTHONPATH=src pytest -q
```

The exact training and evaluation contract is frozen in
[`config/paper-retraining.lock.json`](../config/paper-retraining.lock.json).
Checkpoints and large datasets remain outside Git; their Hugging Face revisions
and local locations are recorded in [`docs/ARTIFACTS.md`](ARTIFACTS.md).
The six final checkpoints and the compact evidence bundle are published at the
immutable model-repository revision
`50ea75ab7972509c2f6827190a965484a2057bcc`.

## Claim boundary

This replication establishes seed stability and paired open-loop improvement on
held-out Isaac Lab demonstrations. It does not establish closed-loop task
success, recovery success after injected failures, or superiority on the
HumanoidArena task suite. Those claims remain blocked until the locked
HumanoidArena runtime passes its upstream OpenDoor contract tests and every
admitted scenario has 20/20 oracle evidence followed by paired closed-loop
baseline and candidate runs.
