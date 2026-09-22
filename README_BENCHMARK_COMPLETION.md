# HRVLA benchmark completion roadmap

This file is the durable execution checklist for completing the frozen
pre-`ours` benchmark. It separates implementation readiness from experimental
evidence. Percentages from different rows must not be averaged because their
units differ.

Last audited: **2026-09-22 (Asia/Bangkok)**. The older CPU-resume notes below
are retained as history; the following GPU audit supersedes their live counters.

## 2026-09-22 recovery step 13–14 result

- Step 13 passed with the corrected PickPlaceBox prompt: **9/9 validated
  runtime traces, 7/7 initial snapshots, and 3/3 failure-start snapshots**.
  The earlier wrong-prompt capture remains preserved but inadmissible.
- Step 14 executed all **180/180** frozen PI0.5 trials, with 180 videos,
  prompt routes verified for all seven task policies, 180 unique locked seeds,
  and zero infrastructure-failure outcomes. The independent result is **82
  successes, 95 timeouts, 3 falls**. Only **2/9 scenarios passed the registered
  20/20 requirement**; seven were rejected behaviorally. See
  `results/benchmark/recovery/oracle_pi05_int8_v0_audit.{json,png}`.
- This is a **completed experiment with a failed admission gate**, not a
  completed nine-scenario benchmark. Do not reroll rejected seeds, weaken the
  threshold after observing outcomes, or promote step 15. A stronger
  independent oracle or changed scenario design requires a new versioned
  protocol and fresh validation; v0 remains available for the paper as a
  negative feasibility result.

## 2026-09-22 dependency decision

The frozen v0 protocol requires all nine scenarios to pass the 20/20 oracle
before compiling the admitted suite. Because seven failed behaviorally, mark
**steps 15–20 rejected by dependency under v0**; this is not a negative result
for the unrun internal methods. Do not spend compute on a dependent step or
silently promote the two passing scenarios as the nine-scenario suite. Steps
22–32 that require the full recovery dataset/lock are also blocked under v0.
Step 21 (GR00T + Isaac coexistence) is independent of recovery admission and
has now passed on CUDA. The external CPU–INT8 sensitivity study is likewise independent
but remains separately scheduled. No step-13 or step-14 rerun is required.

The first step-21 probe was an **invalid implementation attempt**: its batch
JSON used `jobs` while the locked Isaac evaluator requires `episodes`, so no
GR00T inference occurred. Its local logs were retained. After a tested schema
fix on a feature branch, a fresh `step21-coexistence-v2` probe executed one
40-step boxing episode with **two real GR00T inferences** while camera, SONIC,
and Isaac were active. Compute VRAM peaked at **13,011 MiB**, below the frozen
15,500 MiB limit; CUDA is therefore the validated device for this probe.
The 40-step timeout is not a task-success metric. Raw result, telemetry,
video, and reproducible plot are under `results/benchmark/performance/` as
`gr00t_isaac_coexistence_v2*`.

## 2026-09-21 GPU matrix audit and amendment

- The separate INT8 GPU matrix finished **84/84 cells and 1,680/1,680 atomic
  episodes**. All identities are unique, all 168 declared videos exist, and no
  episode has an infrastructure-failure reason. It has not yet passed the
  scientific prompt-route gate.
- The released server did not recognize the exact PickPlaceBox Gym ID. All
  **240 `pp_box` episodes received the literal Gym ID as the language prompt**
  instead of the checkpoint's published `HOI_pp_box` instruction. Their 0/240
  success count is preserved as invalid-route diagnostic evidence, not a valid
  task-policy baseline. The other six task routes were correct.
- An exact-seed Isaac Sim pilot with only the route corrected succeeded at
  step **557**; the original same-seed episode timed out at step **1,450**.
  The pilot is a causal diagnostic, not a result to substitute into the matrix.
- Rerun only the **240 affected PickPlaceBox episodes** into a new directory,
  retain both raw sets, then compile a provenance-explicit amended matrix.
  Never silently overwrite or pool the invalid-route rows.
  The corrected rerun finished **12/12 cells, 240/240 episodes, 145 successes**
  on 2026-09-21. It started as PID `3194009` under Git revision `654c48d`;
  the launch manifest and progress are in
  `_artifacts/HumanoidArena/paper-baselines/pi05-sonic-cuda-int8-ppbox-route-corrected-v1/`.
- The fail-closed amendment compiler passed. It selects the original six
  unaffected tasks and the corrected PickPlaceBox episodes, verifying all
  1,680 identities, 168 videos, source hashes, prompt routes, and the locked
  protocol. The aggregate is **726/1,680 (43.21%)**; the six-task primary
  aggregate excluding diagnostic OpenDoor is **638/1,440 (44.31%)**. The
  versioned JSON and plot are in `results/benchmark/external/`. The original
  wrong-prompt outcomes remain preserved and excluded from these figures.
- The earlier 356 CPU episodes match 356 GPU episodes in model, task, episode
  seed, object seed, and horizon. Their retrospective paired counts are CPU-only
  37 and GPU-only 42 (net +5/356 for INT8). This is a sensitivity check over
  only Boxing and part of DoubleDesk, **not** pre-registered equivalence.
- The corrected recovery capture and frozen oracle result above supersede the
  old wrong-prompt capture and its 0/9 pending counter. Both historical and
  corrected evidence roots remain separate and untouched.

## Executive status

| Workstream | Complete | Expected | Progress | Claim boundary |
| --- | ---: | ---: | ---: | --- |
| Planner component | 5 variants | 5 | 100% | Component evidence; closed-loop comparison still required |
| Recovery component | 6 variants | 6 | 100% | Symbolic fault-injection evidence only |
| Legacy 43-D retraining | 6 seeds | 6 | 100% | Context-only open-loop evidence |
| HumanoidArena 40-D RT training | 3 seeds | 6 | 50% | ST-RT complete; STR-RT waits for recovery data |
| PI0.5 + SONIC external matrix | 84 amended INT8 cells | 84 | Amended 100%; CPU-equivalence unproven | Corrected `pp_box` selected with explicit provenance; broader paired CPU–INT8 sensitivity pending |
| Recovery scenario admission | 9/9 corrected captures; 2/9 scenarios admitted by 180/180 oracle trials | 9 | Capture 100%; oracle experiment 100%; admission 2/9 | Seven frozen behavioral rejections block the registered suite; version a new protocol before further admission trials |
| Internal closed-loop matrix | 0 methods | 5 | 0% | Claim-bearing paired simulation has not started |

The prior 17-cell count was the 2026-09-17 CPU checkpoint, not the current GPU
run. The amended GPU matrix is structurally and prompt-route admitted; this
does not itself establish INT8 equivalence to the original CPU backend.

## Complete benchmark checklist

| # | Gate | Definition of complete | Current state | Remaining work |
| ---: | --- | --- | --- | --- |
| 1 | Scientific protocol lock | Source, simulator, controller, datasets, seeds, splits, metrics, endpoints, and power design are frozen before final outcomes | Complete | Do not change the protocol after observing results |
| 2 | Registered internal comparison | Exactly five rows: GR00T+SONIC, ST, ST-RT, STR, and STR-RT | 5/5 complete | Run all rows on the same frozen plans |
| 3 | Planner component evidence | All five planner/component variants retained | 5/5 complete | Closed-loop evidence remains separate |
| 4 | Recovery component evidence | All six mechanism/fault-injection variants retained | 6/6 complete | Validate in the live simulator |
| 5 | Legacy VLA context | Three ST-RT and three STR-RT legacy 43-D seeds | 6/6 complete | Never present this as the 40-D HumanoidArena rerun |
| 6 | External PI0.5+SONIC matrix | 7 tasks × 4 modes × 3 seeds × 20 rollouts = 84 cells and 1,680 episodes | Amended GPU INT8 1,680/1,680, including corrected `pp_box` 240/240; audit and plot complete | Run broader paired CPU–INT8 sensitivity; do not claim equivalence from existing partial CPU data |
| 7 | Common GR00T adaptation | Train seeds 0/1/2, select one global step on validation, then open hidden evaluation | 3/3 complete; step 300 selected; hidden evaluation complete | Preserve the selected checkpoints and receipts |
| 8 | ST label admission | Audit state/action temporal proxies and adjudicate only rejected episodes under the frozen video-consensus gate | Complete | No threshold changes after outcomes |
| 9 | ST dataset | Materialize and hash-lock 490 train and 70 validation episodes | Complete | Hidden remains locked by protocol |
| 10 | ST-RT training | Three seeds at candidate steps 100/200/300 | 3/3 seeds and 9/9 candidate evaluations complete | No retraining required |
| 11 | ST-RT selection | Select one step without accessing hidden data | Step 300 selected | Validation-only result: common MSE 0.289526, ST-RT MSE 0.288673, paired reduction 0.294% |
| 12 | ST publication | Publish three selected ST-RT checkpoints and the ST dataset at immutable revisions | 4/4 artifacts complete | Receipts are retained in Git |
| 13 | Recovery runtime capture | Validate all nine injectors, seven initial snapshots, and three failure-start snapshots in Isaac Sim | Corrected 9/9 runtime, 7/7 initial, 3/3 failure; manifest passed | Complete; preserve original wrong-prompt capture separately |
| 14 | Independent recovery oracle | Released PI0.5 succeeds 20/20 for each of nine scenarios | 180/180 trials executed and audited; 2/9 scenarios admitted, seven rejected | Admission gate failed; diagnose or propose new versioned protocol rather than reroll behavioral failures |
| 15 | Admitted recovery suite | Capture, source predicate, injector, snapshot, and oracle hashes agree for all scenarios | Rejected by v0 dependency: 2/9 oracle-admitted | Do not compile a nine-scenario suite from partial evidence |
| 16 | Recovery RT dataset | Per scenario: 14 train, 3 validation, 3 hidden episodes | Rejected by v0 dependency: no admitted nine-scenario suite | Do not materialize 126/27/27 episodes without a valid source |
| 17 | Full RT dataset lock | ST and recovery train/validation manifests are canonical and frozen | Rejected by v0 dependency | Recovery manifests unavailable |
| 18 | STR-RT training | Train recovery-conditioned 40-D action decoders for three seeds | Rejected by v0 dependency; 0/3 seeds | Do not train on unadmitted recovery data |
| 19 | STR-RT validation | Evaluate 3 seeds × 3 candidate steps and freeze one global step | Rejected by v0 dependency; 0/9 evaluations | Requires valid STR-RT checkpoints |
| 20 | Final immutable release | Common 3 + ST-RT 3 + STR-RT 3 + ST/recovery datasets = 11 artifacts | Rejected by v0 dependency; 7/11 published | Four recovery-dependent artifacts cannot be promoted |
| 21 | GR00T/Isaac coexistence | One real camera+SONIC+GR00T request is measured under hardware telemetry | Passed on CUDA: 2 real inferences, 13,011 MiB peak versus 15,500 MiB limit; video and plot retained | The 40-step timeout is not task SR; device choice is CUDA for the frozen probe |
| 22 | Frozen checkpoint lock | Nine checkpoint entries bind HF revisions, local manifests, selections, runtime source hashes, and device policy | Blocked under v0 despite independent step 21 | Requires four recovery-dependent artifacts from step 20 |
| 23 | Immutable internal plans | Development, validation, and hidden-final plans have frozen hashes | Blocked under v0; 0/3 | Requires admitted suite and checkpoint lock |
| 24 | Internal development | Five methods × 192 records per method | Blocked under v0; 0/960 records | Requires immutable paired plan |
| 25 | Internal validation | Five methods × 288 records per method | Blocked under v0; 0/1,440 records | Requires development/validation plan |
| 26 | Hidden-final gate | Validation audit is complete and a reviewed `proceed_unchanged` decision is committed | Blocked under v0 | Requires validation evidence |
| 27 | Internal hidden-final | Five methods × 1,248 records per method | Blocked under v0; 0/6,240 records | Requires hidden-final gate |
| 28 | Final metrics | SR, macro/micro RSR, RD, RC, L/H slices, falls, safety, detector P/R/F1, recovery time, policy latency, wall time, utilization, and peak memory | Internal claim blocked under v0; external/recovery diagnostics retained | Compute final paired metrics only from complete audited matrices |
| 29 | Paper statistics | Wilson intervals, paired risk differences, matched odds ratios, exact McNemar tests, Holm correction, per-seed dispersion, and frozen power analysis | Internal claim blocked under v0 | Do not report nonexistent hidden-final comparisons |
| 30 | Visual evidence | Hash-bound representative frames/videos for external, recovery, and internal cells | Partial; internal visuals blocked under v0 | Preserve existing external and recovery video evidence |
| 31 | Final audit | `claim-readiness` and `audit-evidence` both exit zero | Blocked under v0 | Do not claim paper readiness |
| 32 | Benchmark freeze | All evidence is committed and tagged before `ours` exists | Blocked under v0 | Do not tag a complete benchmark or branch `ours` from it |

## Historical CPU-bound handoff (superseded)

The following was the 2026-09-17 operator handoff, not the current plan. The
corrected external matrix and step-13 capture have finished; the step-14
experiment finished with seven behavioral rejections. Do not resume the old
CPU matrix or launch step 15 from this historical command block.

### 1. External PI0.5 + SONIC (historical CPU pause)

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

This chain is currently stopped at the recovery oracle admission gate, despite
all 180 v0 trials having executed. The arrows after that gate are conditional.

## Historical resume commands (do not use for the current gate)

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
