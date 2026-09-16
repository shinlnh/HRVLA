# Benchmark completion status

This file is the persistent source of truth for finishing the pre-`ours`
benchmark. Update it whenever a pipeline stage starts, stops, fails, or reaches
a frozen evidence revision. Do not infer paper readiness from branch names.

Last audited: **2026-09-16 (Asia/Bangkok)**.

## Current state

| Workstream | Finished | Still required for a paper claim |
| --- | --- | --- |
| Planner component | 5/5 algorithm variants, symbolic/Cosmos evaluation | Matched closed-loop rollout of the registered methods |
| Recovery component | 6/6 mechanism variants, symbolic fault injection | Admitted simulator failures and end-to-end recovery rollout |
| VLA retraining | Legacy 43-DoF study: 6/6 training seeds, 42 held-out trajectories × 5 conditions; HumanoidArena 40-D ST-RT/STR-RT: 0/6 seeds | Materialize admitted in-domain datasets, train/select/publish 3 ST-RT + 3 STR-RT checkpoints, then measure closed-loop task and recovery success; legacy open-loop MSE is not SR/RSR |
| External HumanoidArena | Operator-paused at 17/84 evidence-complete seed-cells (15 finalized plus two complete cells inside the interrupted batch); 356/1680 valid episode JSONs at tracked checkpoint | Decide the external runtime path, then resume the locked PI0.5+SONIC matrix without mixing execution backends; treat OpenDoor separately until its upstream contract is fixed |
| Scenario admission | 0/9 scenarios | Immutable snapshot/injector/predicate evidence and independent oracle 20/20 per scenario |
| Internal controlled matrix | 0/5 registered methods; HA state64/action40 train/validation/hidden bridge passed on 700 episodes, the HTTP inference contract is implemented, and the dev/validation/hidden-final split plus power design is frozen | Train the shared bridge and in-domain RT variants, validate real-model servers, then run `gr00t_sonic`, `gr00t_st`, `gr00t_st_rt`, `gr00t_str`, and `gr00t_str_rt` on identical cells |

Recovery admission remains 0/9. Static provenance now resolves and hashes all
7/7 exact upstream task-reward predicates; this corrected several prose-only
definitions that were stronger than the simulator code. Initial/failure
snapshots, nine runtime injectors, and independent 20/20 oracle trials remain
unexecuted and are shown separately in the tracked admission-preflight plot.
All 9/9 static detector/injector contracts are now fully parameterized and
type-checked. Invalid upstream boundaries were removed: DoubleDesk has one
hammer transfer, Boxing terminates at the first hit, and visual navigation has
no waypoint predicate. OpenDoor recovery explicitly enables the upstream
geometry requirement so an obstruction cannot start in an already-successful
state. This static gate does not count as runtime admission.
The hash-addressed Isaac Lab capture/restore implementation is now tested, but
the gate stays at 0/10 until the seven initial and three failure snapshots are
captured from the locked live simulator and independently replayed.
The two action-seam perturbations and deterministic root-velocity primitive are
implemented with strict dimensional checks and explicit per-episode one-shot
reset. They remain runtime-unvalidated until their Isaac Sim traces exist.
Source-backed state detectors now cover all nine locked semantic boundaries;
generic scene primitives cover local-frame velocity, rigid pose placement, and
one-control-step body impulses with exact audit values. Geometry signals reuse
the locked OpenDoor, SitSofa, and Boxing task internals. These implementations
remain at the code/test gate: they do not increment runtime admission before a
live simulator trace proves each detector and perturbation together.
The single-episode upstream wrapper now connects those pieces without modifying
the locked HumanoidArena checkout and records detector/injector/snapshot sidecars.
The fail-closed auditor cross-checks numerical action hashes, physical magnitude,
target, seed, one-shot timing, simulator provenance, and snapshot settle timing.
The full repository suite passes at `163 passed, 16 skipped`; no row is promoted
until the wrapper executes under the real Isaac process after the claim-bearing
external matrix releases CUDA.
The runtime-admission launcher is staged for that handoff: it uses one released
policy load per task, captures all nine scenarios at the locked snapshot seed,
records every run, preserves failed attempts, and refuses GPU overlap. Runtime
success still leaves the independent 20/20 oracle gate closed.
The capture-manifest compiler is also ready and fail-closed: it re-audits the
newest valid attempt, requires 9/9 runtime traces, 7/7 task initial snapshots,
3/3 failure snapshots, and rejects initial-state drift between two scenarios of
the same task. It deliberately reports the oracle gate as 0/9.
The independent-oracle contract is now frozen before observing admission
outcomes: released PI0.5 (independent of the internal GR00T family), 20
hash-derived policy seeds per scenario, identical snapshots/controller/horizon,
and an exact 20/20 pass threshold. Failures are preserved and reject the row;
infrastructure errors must be rerun and cannot be counted as trials.
The same-state execution seam now supports audited readback before policy action:
policy seeds vary independently, while the simulator is reset with the capture
seed and the full declared state hash must match. Online trials restore initial
state then inject once; failure-start trials restore the post-failure state and
never reinject it.
The 180-trial oracle launcher is implemented with 20 episodes per persistent
scenario simulator, one task-shared PI0.5 CPU server, isolated per-trial hashes,
and video for every trial. Resume preserves failed attempt directories. A missed
online trigger is a recorded behavioral failure, not a reroll; any such row
fails the frozen 20/20 gate.
Every runtime trace now binds the exact clean Git implementation revision. The
capture compiler rejects mixed revisions across the nine scenarios, and the
release compiler cannot promote the draft suite unless all snapshots, source
predicates, runtime audits, and independent oracle reports agree exactly. The
draft suite is never edited in place.

The first interrupted external run completed `base_test/boxing/seed-0` and
`seed-1`. It also left 11 valid atomic episode records for seed 2. The optimized
resume completed the remaining nine records, formally closed the third cell,
and entered `semantic/boxing` at 2026-09-15 06:02 UTC without reloading its
task-shared policy server. At the tracked 2026-09-15 09:04 UTC audit it had
produced 109 valid episode records; five seed-cells were complete on disk and
the task/mode batch was still active. The driver-level counter remains lower
until all three seeds in that batch finish. This is explicitly
`partial_non_claim` evidence, not a result table.
At the tracked 2026-09-15 16:39 Asia/Bangkok audit, all three
`semantic/boxing` seed-cells had completed, bringing the immutable snapshot to
120/1,680 episode records and 6/84 cells. The runner then entered
`vision/boxing` without reloading the task-shared PI0.5 checkpoint. This newer
snapshot remains `partial_non_claim`.
The matrix was operator-paused at 162 episodes, shut down cleanly, and resumed
without discarding evidence. At the tracked 2026-09-16 08:41 Asia/Bangkok
audit, all 240 boxing episodes were present and the runner had produced 31
`doubledesk/base_test` episodes. This gives 271/1,680 atomic episode records,
13/84 evidence-complete cells, and 12/84 finalized cells while the current
three-seed batch remains open. Two complete `boxing/vision` summaries missed
between the last episode write and the operator pause were reconstructed only
from their 20/20 atomic JSON records. The runner now reconciles this derived
summary state automatically on every resume.
At 2026-09-16 23:49 Asia/Bangkok the operator paused the external matrix again
to reconsider the CPU-only PI0.5 runtime. The clean shutdown retained 356/1,680
valid atomic episode records: all 240 boxing episodes, all 60
`doubledesk/base_test` episodes, and 56 `doubledesk/semantic` episodes. Seventeen
seed-cells contain 20/20 atomic records; 15 were finalized before the active
three-seed batch was interrupted. No supervisor, policy server, or Isaac Sim
process remains active. Resuming the same backend will submit only the four
missing `doubledesk/semantic` episodes; changing backend requires a separate
homogeneous matrix rather than pooling these records.
The mutable machine progress file is authoritative after this tracked checkpoint.

The common GR00T adaptation inputs and 70/10/20 split are frozen in
`config/humanoidarena-gr00t-training.lock.json`. Three seeds train the same
action-decoder-only profile. A single optimizer step is selected from
100/200/300 by macro-per-task MSE over validation only; the 140 hidden episodes
remain inaccessible until `selection.json` is frozen. The launcher refuses to
start while the external matrix or more than 1 GiB of pre-existing compute VRAM
is present. Separate validation and hidden-evaluation launchers enforce the
exact episode/condition cross-products, refuse partial-output overwrites, and
cryptographically verify the frozen selection before exposing hidden data.

The five-row internal experiment design is frozen in
`config/humanoidarena-internal-protocol.lock.json`. Development uses 4 paired
rollouts per training seed/cell, validation uses 6, and only the 26-rollout
hidden-final split is claim-bearing. With three independently trained seeds,
the final cell contains 78 paired observations. Its predeclared exact McNemar
design has 0.8021 power at the conservative Bonferroni design alpha 0.0125 for
the four Holm-corrected contribution families under `p10=0.26`, `p01=0.05`.
The old 20-rollout design reached only about 0.629 under the same corrected
assumptions, so it was increased before any internal outcomes were observed.
Once the suite is admitted, the immutable plan compiler will produce 192 dev,
288 validation, and 1,248 hidden-final episodes per method; the final table is
therefore 6,240 method-episode records. Splits use disjoint hash-derived seeds.

The real internal rollout path is now implemented without modifying the vendor
evaluator. It composes exact snapshot restore, online/failure-start injection,
source-backed ST transition routing, STR recovery prompts, and a seeded GR00T
HTTP server in one process. Its checkpoint lock is deliberately non-runnable
until all three checkpoint families have three immutable published revisions,
their manifest hashes match local caches, runtime source hashes are frozen, and
the GR00T+Isaac coexistence probe either passes below 15,500 MiB or records an
over-limit measurement and freezes the CPU fallback. The runner shares each
checkpoint server across all applicable methods, batches rollout seeds per
task/scenario, preserves failed attempts, normalizes only audited trials, and
requires an exact complete record set before writing a method audit.
The checkpoint release compiler now hashes the complete behavior-bearing
runtime closure as well: server/wire adapter, state/action bridge, method and
recovery wrappers, semantic detectors, injectors, snapshot restore, plan and
record normalization, planner dependencies, and modality config. A changed
runtime byte therefore invalidates the frozen checkpoint lock even when model
weights themselves are unchanged.
Before any hidden-final outcome existed, the four primary hypothesis endpoints
were bound in the protocol lock: ST vs base and ST-RT vs ST on nominal SR;
STR vs ST and STR-RT vs STR on online-failure RSR. The final reporter requires
78 paired observations in every included task/scenario cell, applies a
two-sided exact McNemar test and Holm correction to exactly these four tests,
and reports paired risk difference, matched odds ratio, per-seed effect
dispersion, Wilson intervals, macro/micro SR/RSR, RD, RC, L/H slices,
detection precision/recall/F1, recovery time, falls, safety, exact policy and
planner request latency, wall time, and hardware telemetry. Failure-start,
OpenDoor, and slice analyses remain explicitly secondary. Runtime timing is
recorded around the existing HTTP call and does not alter policy inputs,
outputs, precision, or control timing.
Hidden-final access is additionally fail-closed behind
`config/humanoidarena-hidden-final.lock.json`. After the complete development
and validation matrices, its compiler re-audits every raw validation record,
binds all five record-file and audit hashes, and permits only an explicit
`proceed_unchanged` decision. The resulting candidate must be reviewed,
committed, and pushed before the hidden-final runner will open its plan.

The first HumanoidArena ST label audit correctly rejected materialization:
state/action proxies still fell back on 69/490 train episodes and 11/70
validation episodes (`0.1408` and `0.1571`, above the frozen `0.05` ceiling).
The failure is localized to semantic OpenDoor transitions plus 11 PickPlaceBox
episodes; lowering the threshold after observing the data is forbidden. The
failed JSON and plot are retained as `subtask_label_audit_v2_failed.*`.
An auditable video adjudicator is now staged for the CUDA handoff. It runs only
on those explicit fallbacks, uses the immutable Cosmos-Reason2-2B revision,
requires three staggered contact-sheet views, minimum per-boundary confidence
`0.70`, maximum cross-view spread `0.08` of episode length, and phases of at
least 40 frames. Rejected or incomplete consensus remains a fallback and keeps
training locked. Its resource gate was exercised successfully: it refuses to
load while the external simulator occupies CUDA. Unit/static tests pass, but
there is no VLM adjudication result yet and therefore no in-domain RT training
claim.
The two observed-data freeze points are now executable without hand-editing
JSON: `compile_humanoidarena_rt_lock.py adjudication` accepts only the original
model/threshold policy and a complete passing report; `datasets` accepts only
490/70 ST episodes and 126/27 recovery episodes with canonical manifests. Each
writes a candidate rather than replacing the tracked lock. Training remains
blocked until the reviewed candidate is committed on `main`.

![Readiness by independent workstream](results/benchmark/readiness/benchmark_readiness.png)

## Frozen scope before `ours`

The benchmark is complete only when every applicable box below is checked. This
is deliberately finite: adding more unrelated papers is not part of the exit
condition.

- [x] Lock upstream source, model, dataset, controller, and simulator revisions.
- [x] Register the five internal ablation rows and their exact component changes.
- [x] Retain raw planner, recovery, multi-seed retraining, Isaac Sim, video, and
  hardware evidence with honest claim boundaries.
- [ ] Publish the new HumanoidArena common/ST-RT/STR-RT selected checkpoints and
  in-domain datasets to immutable Hugging Face revisions; keep their hashes in
  Git. The already-published legacy 43-DoF artifacts do not satisfy this gate.
- [ ] Finish all 84 external PI0.5+SONIC cells: 7 tasks × 4 modes × 3 seeds × 20
  rollouts. The six upstream-valid tasks form the primary aggregate. OpenDoor is
  a diagnostic/appendix row unless its two locked upstream contract tests pass
  before results are inspected.
- [x] Implement and validate the HumanoidArena 40-D action bridge for the five
  GR00T/SONIC internal methods. The existing Arena G1 retrains use a different
  43-DoF/direct-action contract and cannot be presented as HumanoidArena reruns.
- [ ] Pass the frozen ST temporal-label admission gate, materialize train and
  validation views, and preserve the rejected-attempt audits. Video consensus
  is training supervision only and must not be reported as simulator success.
- [ ] Materialize the nine recovery scenarios with deterministic semantic-event
  injectors, snapshots, predicates, and exact hashes.
- [ ] Pass independent oracle admission at 20/20 for every included scenario.
- [x] Freeze development/validation/hidden-final seed derivation and an
  independently checked power design before creating or tuning `ours`.
- [ ] Materialize the three immutable plan hashes from the admitted suite before
  creating or tuning `ours`; the compiler rejects draft scenarios.
- [ ] Run all five internal methods on the identical plan with 3 training seeds
  and 26 hidden-final rollout seeds per cell. Never pool `nominal`,
  `failure_start`, and
  `online_failure`.
- [ ] Report SR, macro RSR, RD, RC, L-level/H-axis slices, fall and safety rates,
  detection precision/recall/F1, recovery time, policy latency, wall time,
  CPU/GPU utilization, and peak memory.
- [ ] Attach Wilson intervals, paired McNemar tests, Holm correction, effect
  sizes, and a power analysis. Increase repeats if the predeclared power target
  is not met.
- [ ] Make `claim-readiness` and `audit-evidence` exit zero; verify every artifact
  hash and retain representative success/failure videos and Isaac Sim frames.
- [ ] Tag the frozen pre-`ours` state as `benchmark/hrvla-v1-frozen`. Create the
  future `ours` branch only from that tag.

## Comparison table contract

| Row | What is compared directly | Evidence status |
| --- | --- | --- |
| PI0.5 + SONIC | Released HumanoidArena task checkpoints under the locked 84-cell protocol | Running; partial non-claim |
| GR00T + SONIC | No subtask planner, no recovery, no project retraining | Registered; closed-loop pending |
| GR00T-ST | Adds the subtask planner only | Registered; closed-loop pending |
| GR00T-ST-RT | Adds in-domain post-training to ST | New 40-D in-domain training pending; legacy 43-DoF open-loop evidence is context only |
| GR00T-STR | Adds transition-aware recovery to ST | Component complete; closed-loop pending |
| GR00T-STR-RT | Adds recovery-conditioned post-training to STR | New 40-D in-domain training pending; legacy 43-DoF open-loop evidence is context only |
| `ours` | Future contribution against the frozen rows above | Must not start until the frozen tag exists |

τ0-VLA, Vesta, Anticipation-VLA, STEP Planner, Hi Robot, BATON, AgentChord,
ReKep, DoReMi, Inner Monologue, ReSYNC, Dream2Fix, LIBERO-RECOVER, Pro-HOI,
OmniContact, SUGAR, and HumanoidExo remain paper-reported context unless their
official artifacts are rerun under the same observation/action/controller and
episode contract. A proxy implementation must never be labeled an official
paper rerun.

## Live pipeline and reproducible commands

The optimized external runner keeps one exact CPU PI0.5 server per task and one
persistent Isaac Sim process per mode. It resumes finished episode IDs, records
one representative video per ten scheduled episodes, disables diagnostic tensor
dumps, and captures five-second hardware telemetry. The scientific contract is
unchanged; only orchestration and non-metric I/O differ.

```bash
python3 -u scripts/run_humanoidarena_baseline_matrix_fast.py \
  --cpu-threads 24 --compile-threads 32 --interop-threads 2

python3 scripts/summarize_humanoidarena_baseline_matrix.py --allow-partial
python3 scripts/render_benchmark_evidence.py
PYTHONPATH=src python3 -m pytest -q

# After the external matrix releases CUDA, resolve only rejected training-label
# proxies. Inspect the report before freezing its audit hash in the RT lock.
_vendor/Isaac-GR00T/.venv/bin/python \
  scripts/run_humanoidarena_subtask_video_adjudication.py
python3 scripts/compile_humanoidarena_rt_lock.py adjudication
# Review/commit the candidate lock, then materialize ST train + validation.
python3 scripts/prepare_humanoidarena_subtask_data.py \
  --audit-only \
  --audit-output results/benchmark/retraining/subtask_label_audit_v3.json \
  --audit-plot results/benchmark/retraining/subtask_label_fallback_v3.png
# After ST and recovery train/validation views exist:
python3 scripts/compile_humanoidarena_rt_lock.py datasets

# After 9/9 capture and 9/9 independent oracle admission:
python3 scripts/compile_humanoidarena_admitted_suite.py
python3 scripts/prepare_humanoidarena_internal_plans.py

# After all three checkpoint families are trained/selected and both in-domain
# datasets have frozen manifests, stage and publish the content-addressed set:
python3 scripts/release_humanoidarena_artifacts.py stage
_vendor/Isaac-GR00T/.venv/bin/python \
  scripts/release_humanoidarena_artifacts.py publish
# Run one real camera/SONIC/GR00T request concurrently under telemetry. The
# probe automatically validates CUDA or records and validates the CPU fallback.
_vendor/Isaac-GR00T/.venv/bin/python \
  scripts/probe_humanoidarena_gr00t_coexistence.py \
  --checkpoint _artifacts/retraining/humanoidarena-common/seed-0/checkpoints/checkpoint-<selected-step> \
  --manifest _artifacts/HumanoidArena/release/humanoidarena-v1/manifests/common-seed-0.json
# After the exclusive probe writes coexistence-probe.json:
python3 scripts/release_humanoidarena_artifacts.py compile-lock
# Review the candidate, then commit it as the tracked checkpoint lock before
# compiling or executing any internal plan.

python3 scripts/run_humanoidarena_internal_matrix.py --split development --dry-run
python3 -u scripts/run_humanoidarena_internal_matrix.py --split development
python3 -u scripts/run_humanoidarena_internal_matrix.py --split validation
python3 scripts/compile_humanoidarena_hidden_final_gate.py
# Review and commit the candidate as config/humanoidarena-hidden-final.lock.json;
# only then expose these one-shot outcomes:
python3 -u scripts/run_humanoidarena_internal_matrix.py --split hidden_final
python3 scripts/render_humanoidarena_internal_results.py
```

During execution, machine-live progress is written to
`_artifacts/HumanoidArena/paper-baselines/pi05-sonic/progress.json`; hardware
samples are appended to `hardware-telemetry.jsonl`. Both are ignored while
mutable. At a milestone or clean completion, regenerate the compact tracked
summary and plots under `results/`, update this file, commit, and push `main`.

A durable post-external supervisor is installed at
`scripts/run_humanoidarena_post_external_pipeline.py`. It waits for both the
1680/1680 completion marker and driver exit, resumes the external matrix up to
three times after an unexpected exit, then runs only pre-frozen independent
work: complete-matrix audit; common GR00T three-seed training, validation
selection, and hidden open-loop evaluation; Cosmos temporal adjudication;
recovery runtime capture, manifest compilation, 20/20 oracle admission,
admitted-suite compilation, and recovery-dataset materialization. Each command,
exit code, dependency block, and timestamp is persisted under
`_artifacts/HumanoidArena/benchmark-pipeline/`. It deliberately stops before
editing tracked locks or starting RT training, because those steps require an
audited human/agent freeze of newly observed hashes. It never promotes partial
results or edits a claim-bearing file.

A second durable supervisor,
`scripts/run_humanoidarena_completion_supervisor.py`, waits on that handoff and
continues every authorized stage through RT training/publication, the
coexistence probe, development, validation, hidden-final audit, and final
plots. At each observed-data boundary it writes a candidate and waits for the
reviewed lock to appear on `main`; it never edits or commits a tracked lock
itself. Its resumable status and logs live beside the first supervisor under
`_artifacts/HumanoidArena/benchmark-pipeline/`.
The final evidence bundle refuses an incomplete visual matrix. With up to 24
CPU workers it extracts three timestamped frames from one hash-bound video for
each of 28 external task/mode cells, nine recovery scenarios, and 15 internal
method/protocol cells. The 52 contact sheets and their SHA-256 manifest are
written under `results/humanoidarena/video-evidence/` for Git, while the full
videos remain large artifacts rather than being duplicated into the source
repository.

## Resource rule

The RTX 5070 Ti has 16,303 MiB VRAM. Isaac Sim plus SONIC uses roughly 6–7 GiB,
while the exact float32 PI0.5 policy parameters alone use 9,353,976,928 bytes,
before activations. Running both fully on the GPU has no safe memory margin.
Therefore the locked baseline keeps PI0.5 on CPU and reserves CUDA for physics,
rendering, and SONIC. Quantization or reduced precision would be a different
baseline and requires a separately labeled ablation. “Maximum hardware” means
maximum safe throughput under the unchanged scientific method, not an unsafe
utilization target.

A fixed-input steady-state profile on the i9-14900K measured median PI0.5
request latency of 23.72s, 29.18s, 17.52s, and 40.24s at 8, 16, 24, and 32
intra-op threads respectively. The pipeline therefore uses 24 policy threads;
this is 2.30× faster than forcing all 32 logical threads. The remaining cores
serve Isaac Sim, I/O, and orchestration. See the
[profile JSON](results/benchmark/performance/pi05_cpu_thread_profile.json) and
[thread-scaling plot](results/benchmark/performance/pi05_cpu_thread_scaling.png).

## Stop/restart log

- 2026-09-15: stopped legacy PID `331152` and its benchmark-only process group
  after SIGTERM did not drain children. No unrelated process was killed.
- 2026-09-15: retained all 51 finished episode JSONs and their videos; no partial
  result was promoted to a claim.
- 2026-09-15: replaced cell-local server startup with task-shared server
  orchestration to reduce policy loads from at most 84 to 7 and simulator starts
  from at most 84 to 28 for a fresh full run.
- 2026-09-15 05:28 UTC: started the optimized full-matrix resume as PID `1551582`
  with 24 policy threads, 32 compile threads, two inter-op threads, and five-second
  telemetry. After the first valid new result, the driver, policy server, and
  simulator were raised from inherited `nice=5` to normal `nice=0`; future child
  processes inherit the corrected priority. No unrelated process was reprioritized.
- 2026-09-15 06:02 UTC: completed the resumed `base_test/boxing/seed-2` cell and
  entered `semantic/boxing`. The task-shared PI0.5 server remained alive across
  the mode boundary, validating the intended reload-elimination path.
- 2026-09-15 09:39 UTC: completed all three `semantic/boxing` cells at
  120/1,680 episodes and entered `vision/boxing`, again retaining the same
  task-shared PI0.5 server.
- 2026-09-15 12:07 UTC: operator-paused at 162/1,680 episodes by stopping the
  two supervisors before the matrix runner. PI0.5 and Isaac exited cleanly;
  no completed episode was removed.
- 2026-09-15 15:21 UTC: resumed from the atomic evidence with only 18 missing
  `boxing/vision` episodes. The runner finished all boxing modes and entered
  `doubledesk/base_test` without replaying a completed episode.
- 2026-09-16 01:41 UTC: tracked 271/1,680 episodes. Reconciled two missing
  derived cell summaries from complete 20/20 atomic JSON sets and added a
  regression-tested startup reconciliation path for future operator pauses.
- 2026-09-16 16:49 UTC: operator-paused all three pipeline layers at
  356/1,680 episodes. Supervisors were stopped before the matrix runner; the
  runner drained PI0.5 and Isaac, GPU allocation returned to the desktop-only
  baseline, and no completed episode was removed.

See [the local storage cleanup ledger](docs/LOCAL_STORAGE_CLEANUP.md),
[the benchmark protocol](docs/BENCHMARK.md), and
[the artifact contract](docs/ARTIFACTS.md). The internal action-interface work
is tracked in [the HumanoidArena 40-D GR00T bridge note](docs/HUMANOIDARENA_GR00T_BRIDGE.md).
