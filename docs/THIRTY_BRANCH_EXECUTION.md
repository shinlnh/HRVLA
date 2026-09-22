# Thirty-branch execution ledger (v2)

Updated: 2026-09-22. This is the canonical **work ledger**, not a results table.
The remote contract remains ten architecture branches, twenty benchmark branches,
plus `dev` and `main`. Do not make a new branch for a run, retry, or failed result.
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
| PI05 | partial: released inference backend and SONIC interface | frozen checkpoint/backend and paired closed-loop smoke |
| PI05-ST | partial: model-preserving planner routing | connect PI0.5 HTTP client to observed scene detectors |
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
means SIMPLE. Every row is currently **PENDING**: branch reservation and old
component evidence are not new v2 rollouts.

| Architecture token | HA branch | SP branch | Immediate short task |
| --- | --- | --- | --- |
| GR00T | pending | pending | sync architecture tip; freeze suite adapter |
| GR00T-ST | pending | pending | sync architecture tip; verify detector contract |
| GR00T-STR | pending | pending | sync architecture tip; verify failure contract |
| GR00T-ST-RT | pending | pending | sync architecture tip; require checkpoint selection |
| GR00T-STR-RT | pending | pending | sync architecture tip; require recovery checkpoint |
| PI05 | pending | pending | sync architecture tip; freeze precision/backend |
| PI05-ST | pending | pending | sync architecture tip; verify HTTP planner hook |
| PI05-STR | pending | pending | sync architecture tip; verify recovery injector |
| PI05-ST-RT | pending | pending | sync architecture tip; require trained v3 checkpoint |
| PI05-STR-RT | pending | pending | sync architecture tip; require recovery-trained checkpoint |

## Ordered work queue

1. Audit the 32 remote heads and preserve any dirty worktree. Keep `dev/main`
   unchanged until the later `ours` contribution is selected.
2. Fast-forward each of the 20 benchmark branches' code ancestry to its current
   architecture tip via a normal merge commit; retain its own PENDING README.
3. Add a branch-specific, fail-closed evaluation preflight and test it on all
   twenty branches. A preflight pass is not an evaluation result.
4. Pin the two public suites, G1/SONIC observation/action contracts, task lists,
   initial snapshots, seeds, horizons, success/fall predicates, and precision.
   SIMPLE needs a separately validated adapter: its published baseline scores
   are context, not this project's scores.
5. Complete observed scene detectors and recovery injectors; run short
   Isaac Sim smoke tests on each architecture without scoring paper results.
6. Convert/prepare PI0.5 and GR00T in-domain RT datasets without modifying
   source data; audit split leakage and recovery-label provenance.
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

None started under this v2 ledger yet. The first expected long job is an
in-domain PI0.5 ST-RT dataset conversion/train or a paired HA simulator
matrix, after their preflight and resource checks pass.
