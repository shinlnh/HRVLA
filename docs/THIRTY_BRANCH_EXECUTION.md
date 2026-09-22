# Thirty-branch execution ledger (v2)

Updated: 2026-09-22. This is the canonical **work ledger**, not a results table.
GitHub has exactly ten architecture branches, twenty benchmark branches, plus
`dev` and `main` (32 heads). The HELIOS working repository additionally keeps
17 local historical branches; those are not GitHub heads and must not be
confused with new matrix rows. Do not make a new branch for a run, retry, or
failed result.
Preserve v0 evidence separately; never promote it into a v2 cell by renaming it.

## Completion gates

An architecture branch is `code-ready` only when its VLA/checkpoint, SONIC
state64/action40 adapter, planner/recovery hooks (if applicable), task detectors,
and an end-to-end simulator smoke test are all pinned and tested. An RT branch
also needs an immutable, validation-selected in-domain checkpoint. Unit tests
or a launchable training command alone mean `partial`, not `code-ready`.

A benchmark branch is `complete` only when it inherits the corresponding
architecture implementation and has a frozen public-suite revision, paired
episode plan, exact model/controller hashes, raw per-episode results including
failures, nominal/failure-start/online-failure metrics where applicable,
plots/video evidence, and an audit of the expected episode set. A failed method
is a recorded outcome, not a reason to skip another row. Do not run a hidden
split before validation selection is frozen.

## Architecture ledger (10)

| Token | Code state | Next gate |
| --- | --- | --- |
| GR00T | partial: 64D/40D bridge and HTTP server | pin common checkpoint; closed-loop Isaac Sim smoke |
| GR00T-ST | partial: observed-state planner routing | HA/SIMPLE scene detectors and live subtask transitions |
| GR00T-STR | partial: bounded recovery routing | real failure detector, snapshot/rollback, live recovery smoke |
| GR00T-ST-RT | partial: guarded 40D training path | select/pin 3-seed 40D checkpoint; closed-loop smoke |
| GR00T-STR-RT | partial: guarded recovery-labelled training path | admissible 40D recovery data, train/select/pin, live smoke |
| PI05 | code-ready for HA: seven checkpoint/SONIC locks and full closed-loop baseline audited | SIMPLE-compatible adapter remains separate |
| PI05-ST | partial: 7/7 bounded live HA integration smokes at one revision | full-horizon paired ST results and observed live transition |
| PI05-STR | partial: bounded recovery instruction routing | connect nine failure injectors and validate live trajectories |
| PI05-ST-RT | partial: guarded LeRobot v3 trainer | convert audited v2.1 copy; train/select/pin checkpoint |
| PI05-STR-RT | partial: guarded recovery-labelled v3 trainer | collect/verify labels, train/select/pin checkpoint |

The 43-DoF historical GR00T Arena checkpoints are not interchangeable with
HumanoidArena's 40-D action contract. The PI0.5 CUDA INT8 v1 matrix is not a
matched ten-method v2 result. Record precision/backend sensitivity if any
method uses a different policy execution path.

## Benchmark ledger (20)

For each token below, both `feat(<token>-benchmark-HA)/evaluate` and
`feat(<token>-benchmark-SP)/evaluate` exist. `HA` means HumanoidArena; `SP`
means SIMPLE. Each branch now contains its architecture tip, a pinned public
suite revision, a method-specific v2 contract, and a fail-closed preflight.
The PI05 HA branch is now `complete` as a **baseline-only six-task result**:
1,440/1,440 primary episodes and 144/144 sampled primary videos passed the
raw-evidence recompilation, checkpoint, controller, seed, and visual audits.
OpenDoor's 240 episodes remain diagnostic. The other 19 benchmark branches
remain `pending`; no paired ten-method comparison is complete. This baseline
was retrospectively frozen and must not be described as preregistered.

| Architecture token | HA branch | SP branch | Immediate short task |
| --- | --- | --- | --- |
| GR00T | pending | pending | pin compatible base model and closed-loop SONIC smoke |
| GR00T-ST | pending | pending | observed scene detector and live planner transitions |
| GR00T-STR | pending | pending | failure injector, rollback, live recovery smoke |
| GR00T-ST-RT | pending | pending | validated 40-D in-domain checkpoint selection |
| GR00T-STR-RT | pending | pending | admissible recovery data and checkpoint selection |
| PI05 | complete baseline only (638/1,440 primary successes) | pending | pair later methods to the frozen HA identities; validate SIMPLE adapter |
| PI05-ST | pending | pending | observed detector and live PI0.5 planner hook |
| PI05-STR | pending | pending | nine failure injectors and live recovery smoke |
| PI05-ST-RT | pending | pending | safe v3 ST data, train and select checkpoint |
| PI05-STR-RT | pending | pending | recovery labels, safe v3 data, checkpoint |

Pinned public source revisions: HumanoidArena
`68479287a784a69be9ce6ad739311d2f11f75ef9`; SIMPLE
`6d10628794d9c7de4596b4f2afb2c054a637c2bc`. SIMPLE's official SONIC
path consumes 43-D state and emits 30 x 78-D action chunks, whereas the
current HumanoidArena policy bridge uses 64-D state and 40-D action. All ten
SP preflights explicitly reject that mismatch. A dimension label alone is
not an adapter: use a validated scene/state/action map or train a separate
SIMPLE-compatible checkpoint, then rerun paired smokes before scoring.

## Ordered work queue

1. **Done:** audit the 32 GitHub heads and preserve dirty worktrees. Keep
   `dev/main` unchanged until the later `ours` contribution is selected.
2. **Done:** merge each of the 20 benchmark branches with its pinned
   architecture tip; retain its own PENDING README.
3. **Done:** add a branch-specific, fail-closed evaluation preflight, run five
   unit tests per branch, and verify every pending branch refuses a rollout.
   A preflight pass is not an evaluation result.
4. **Partial:** pin the two public source revisions. Still freeze task lists,
   G1/SONIC adapter, initial snapshots, seeds, horizons, success/fall predicates,
   and precision. SIMPLE needs a separately validated adapter/checkpoint;
   its published baseline scores are context, not this project's scores.
5. Complete observed scene detectors and recovery injectors; run short
   Isaac Sim smoke tests on each architecture without scoring paper results.
6. Convert/prepare PI0.5 and GR00T in-domain RT datasets without modifying
   source data; audit split leakage and recovery-label provenance.
   **PI0.5 ST-RT blocker:** the available 490-episode/380,927-frame training
   set is LeRobot v2.1, has packed-video symlinks and per-episode frame offsets,
   and lacks `meta/episodes_stats.jsonl`. The upstream in-place v2.1-to-v3
   converter assumes complete per-episode videos, so it cannot safely convert
   this set. Build and validate a non-destructive packed-video-aware v3 export
   before any training. Do not point the existing training launcher at v2.1.
7. Train RT candidates with resume-safe logs, select checkpoints **only on
   validation**, and publish immutable hashes. Do not train all variants from
   one unqualified checkpoint merely to fill the matrix.
8. Freeze one paired HA protocol; run and audit ten rows independently.
9. Freeze one paired SP protocol; run and audit ten rows independently.
10. Produce comparison plots, uncertainty intervals, failure-case/video index,
    and a single paper-readiness audit. Choose the future `ours` parent only
    from validation evidence; then, and only then, create/merge `ours`.

When a multi-hour train or simulator matrix begins, record its exact command,
PID/job ID, output directory, start timestamp, resume command, and progress
check here. Stop interactive supervision and let the run continue; never mark
the branch complete just because the process exited.

## Long-running job

None started under this v2 ledger yet. The first long stage is building and
validating the packed-video-aware PI0.5 ST-RT v3 dataset, then training, or a
paired HA simulator matrix once its protocol locks pass. The existing v2.1
dataset must not be fed to an incompatible training path to keep hardware busy.
