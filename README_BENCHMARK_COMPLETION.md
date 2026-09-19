# HRVLA benchmark completion roadmap

This file is the durable execution checklist for completing the frozen
pre-`ours` benchmark. It separates implementation readiness from experimental
evidence. Percentages from different rows must not be averaged because their
units differ.

Last audited: **2026-09-17 (Asia/Bangkok)** at Git revision `1fce079`.

## Executive status

| Workstream | Complete | Expected | Progress | Claim boundary |
| --- | ---: | ---: | ---: | --- |
| Planner component | 5 variants | 5 | 100% | Component evidence; closed-loop comparison still required |
| Recovery component | 6 variants | 6 | 100% | Symbolic fault-injection evidence only |
| Legacy 43-D retraining | 6 seeds | 6 | 100% | Context-only open-loop evidence |
| HumanoidArena 40-D RT training | 3 seeds | 6 | 50% | ST-RT complete; STR-RT waits for recovery data |
| PI0.5 + SONIC external matrix | 17 evidence-complete cells | 84 | 20.24% | Partial non-claim until the locked matrix is complete |
| Recovery scenario admission | 0 scenarios | 9 | 0% | Static contracts pass; runtime/oracle evidence is absent |
| Internal closed-loop matrix | 0 methods | 5 | 0% | Claim-bearing paired simulation has not started |

The external driver has finalized 15 cells, while two additional cells contain
20/20 valid atomic episode records. The filesystem therefore has 17
evidence-complete cells even though the mutable driver counter is 15.

## Complete benchmark checklist

| # | Gate | Definition of complete | Current state | Remaining work |
| ---: | --- | --- | --- | --- |
| 1 | Scientific protocol lock | Source, simulator, controller, datasets, seeds, splits, metrics, endpoints, and power design are frozen before final outcomes | Complete | Do not change the protocol after observing results |
| 2 | Registered internal comparison | Exactly five rows: GR00T+SONIC, ST, ST-RT, STR, and STR-RT | 5/5 complete | Run all rows on the same frozen plans |
| 3 | Planner component evidence | All five planner/component variants retained | 5/5 complete | Closed-loop evidence remains separate |
| 4 | Recovery component evidence | All six mechanism/fault-injection variants retained | 6/6 complete | Validate in the live simulator |
| 5 | Legacy VLA context | Three ST-RT and three STR-RT legacy 43-D seeds | 6/6 complete | Never present this as the 40-D HumanoidArena rerun |
| 6 | External PI0.5+SONIC matrix | 7 tasks × 4 modes × 3 seeds × 20 rollouts = 84 cells and 1,680 episodes | 356/1,680 episodes; 17/84 evidence-complete cells | Resume the exact CPU PI0.5 backend for the remaining 1,324 episodes; do not pool a different backend |
| 7 | Common GR00T adaptation | Train seeds 0/1/2, select one global step on validation, then open hidden evaluation | 3/3 complete; step 300 selected; hidden evaluation complete | Preserve the selected checkpoints and receipts |
| 8 | ST label admission | Audit state/action temporal proxies and adjudicate only rejected episodes under the frozen video-consensus gate | Complete | No threshold changes after outcomes |
| 9 | ST dataset | Materialize and hash-lock 490 train and 70 validation episodes | Complete | Hidden remains locked by protocol |
| 10 | ST-RT training | Three seeds at candidate steps 100/200/300 | 3/3 seeds and 9/9 candidate evaluations complete | No retraining required |
| 11 | ST-RT selection | Select one step without accessing hidden data | Step 300 selected | Validation-only result: common MSE 0.289526, ST-RT MSE 0.288673, paired reduction 0.294% |
| 12 | ST publication | Publish three selected ST-RT checkpoints and the ST dataset at immutable revisions | 4/4 artifacts complete | Receipts are retained in Git |
| 13 | Recovery runtime capture | Validate all nine injectors, seven initial snapshots, and three failure-start snapshots in Isaac Sim | 0/9 runtime; 0/7 initial; 0/3 failure snapshots | Execute the live recovery capture workload |
| 14 | Independent recovery oracle | Released PI0.5 succeeds 20/20 for each of nine scenarios | 0/9 scenarios and 0/180 trials | Run all trials; retain behavioral failures and never reroll them |
| 15 | Admitted recovery suite | Capture, source predicate, injector, snapshot, and oracle hashes agree for all scenarios | 0/9 admitted | Compile the immutable admitted suite after gates 13–14 |
| 16 | Recovery RT dataset | Per scenario: 14 train, 3 validation, 3 hidden episodes | 0/180 episodes | Materialize 126 train, 27 validation, and 27 hidden episodes |
| 17 | Full RT dataset lock | ST and recovery train/validation manifests are canonical and frozen | ST frozen; recovery manifests null | Compile, review, and commit the full candidate lock |
| 18 | STR-RT training | Train recovery-conditioned 40-D action decoders for three seeds | 0/3 seeds | Train steps 100/200/300 after the recovery dataset opens |
| 19 | STR-RT validation | Evaluate 3 seeds × 3 candidate steps and freeze one global step | 0/9 evaluations | Retain JSON, per-seed data, and plots |
| 20 | Final immutable release | Common 3 + ST-RT 3 + STR-RT 3 + ST/recovery datasets = 11 artifacts | 7/11 published and verified | Publish three STR-RT checkpoints and the recovery dataset |
| 21 | GR00T/Isaac coexistence | One real camera+SONIC+GR00T request is measured under hardware telemetry | Pending | Use CUDA only if measured peak is at most 15,500 MiB; otherwise freeze the validated CPU fallback |
| 22 | Frozen checkpoint lock | Nine checkpoint entries bind HF revisions, local manifests, selections, runtime source hashes, and device policy | Pending | Compile, review, and commit after release and coexistence probe |
| 23 | Immutable internal plans | Development, validation, and hidden-final plans have frozen hashes | 0/3 | Compile only from the admitted suite and checkpoint lock |
| 24 | Internal development | Five methods × 192 records per method | 0/960 records | Execute identical paired plans; development is not the final claim |
| 25 | Internal validation | Five methods × 288 records per method | 0/1,440 records | Audit all raw records before opening hidden-final |
| 26 | Hidden-final gate | Validation audit is complete and a reviewed `proceed_unchanged` decision is committed | Pending | Compile, review, and commit the candidate gate |
| 27 | Internal hidden-final | Five methods × 1,248 records per method | 0/6,240 records | Execute the one-shot claim-bearing matrix |
| 28 | Final metrics | SR, macro/micro RSR, RD, RC, L/H slices, falls, safety, detector P/R/F1, recovery time, policy latency, wall time, utilization, and peak memory | Reporting code ready; evidence absent | Compute only from complete audited matrices |
| 29 | Paper statistics | Wilson intervals, paired risk differences, matched odds ratios, exact McNemar tests, Holm correction, per-seed dispersion, and frozen power analysis | Protocol frozen; outcomes absent | Render final tables and plots after hidden-final |
| 30 | Visual evidence | Hash-bound representative frames/videos for external, recovery, and internal cells | Partial | Produce the final 52 contact sheets and their SHA-256 manifest |
| 31 | Final audit | `claim-readiness` and `audit-evidence` both exit zero | Pending | Verify every result, artifact, revision, manifest, video, and plot |
| 32 | Benchmark freeze | All evidence is committed and tagged before `ours` exists | Pending | Create `benchmark/hrvla-v1-frozen`, then branch `ours` from that tag |

## CPU-bound work deliberately left for last

### 1. External PI0.5 + SONIC

- Paused safely at 356/1,680 valid episodes.
- Seventeen cells contain 20/20 atomic records; 15 are finalized by the driver.
- The interrupted batch is `doubledesk/semantic`.
- Resume keeps every completed episode and submits only missing IDs.
- The six upstream-valid tasks form the primary aggregate.
- OpenDoor remains a diagnostic/appendix row unless its two locked upstream
  contract tests pass before results are inspected.

### 2. Recovery admission package

- Nine live runtime captures.
- Seven initial snapshots and three failure-start snapshots.
- 180 independent PI0.5 oracle trials.
- Admitted-suite compilation.
- Recovery dataset materialization: 126 train, 27 validation, 27 hidden.

No downstream STR-RT or internal closed-loop claim can open before the recovery
package passes.

## Required execution order

```text
External matrix 84/84
  -> Recovery capture 9/9
  -> Recovery oracle 180/180
  -> Admitted recovery suite
  -> Recovery dataset
  -> Full RT dataset lock
  -> STR-RT training 3/3
  -> STR-RT validation and selection
  -> Publish four remaining artifacts
  -> GR00T/Isaac coexistence probe
  -> Frozen checkpoint lock
  -> Development/validation/hidden-final plans
  -> Internal development matrix
  -> Internal validation matrix
  -> Hidden-final gate
  -> Internal hidden-final matrix
  -> Final metrics, statistics, plots, and videos
  -> claim-readiness and audit-evidence
  -> Tag benchmark/hrvla-v1-frozen
  -> Create the future ours branch
```

## Resume commands

Run the remaining CPU-bound work package when the machine is available:

```bash
cd /HELIOS/Robotics/HRVLA
python3 -u scripts/run_humanoidarena_post_external_pipeline.py --poll-seconds 60
```

The completion supervisor may wait in a second terminal. It uses effectively
zero compute while waiting and resumes only after the post-external handoff and
reviewed lock commits:

```bash
cd /HELIOS/Robotics/HRVLA
python3 -u scripts/run_humanoidarena_completion_supervisor.py --poll-seconds 60
```

Live external progress:

```bash
cd /HELIOS/Robotics/HRVLA
watch -n 30 'python3 -m json.tool _artifacts/HumanoidArena/paper-baselines/pi05-sonic/progress.json'
```

Repository tests and evidence rendering:

```bash
cd /HELIOS/Robotics/HRVLA
PYTHONPATH=src python3 -m pytest -q
python3 scripts/render_benchmark_evidence.py
```

## Retained evidence and storage state

- All three common and all three ST-RT selected checkpoints are published.
- The 106.94 GB logical ST dataset is published and remotely verified.
- Seven pre-recovery release receipts are tracked under
  `results/benchmark/release/pre-recovery/`.
- Selected checkpoint-300 directories remain local.
- Twelve unselected checkpoint directories and twelve redundant root shards
  were removed only after immutable receipt verification.
- Exact reclaimed space: 106.76 GiB.
- Free local storage at this audit: approximately 340 GiB.
- Current full test result: 204 passed, 17 skipped.

See also:

- `BENCHMARK_STATUS.md` for the chronological source of truth.
- `docs/BENCHMARK.md` for the scientific protocol.
- `docs/ARTIFACTS.md` for the artifact contract.
- `results/benchmark/readiness/benchmark_readiness.png` for the current visual
  summary.
