# Paper matrix v2: two VLA families, five architectures, two benchmarks

This is the project goal and branch contract requested on 2026-09-22. It is a
plan, **not** a claim that unimplemented architectures or unrun benchmarks are
complete. Preserve all v0 results and failed trials as historical evidence.
Do not rewrite `main` or `dev` to make a missing result appear complete.

Current execution state is tracked in
[`THIRTY_BRANCH_EXECUTION.md`](THIRTY_BRANCH_EXECUTION.md). The original migration
checkpoint had six architecture branches and no v2 benchmark branches; the
current remote has all 10 architecture and 20 benchmark branches plus `dev`
and `main`. This count measures organization, not experimental completion.

## Architecture branches (10)

Each architecture branch owns a runnable method and its tests, not a historical
phase of the project. `ST` adds sub-task planning; `STR` adds recovery on top of
ST; `ST-RT` retrains the ST VLA; `STR-RT` retrains the recovery VLA.

| Family | Variant | Architecture branch | State at this plan |
| --- | --- | --- | --- |
| GR00T+SONIC | base | `feat(GR00T)/reproduce-gr00t-sonic-whole-body` | Existing branch; task-level paper matrix pending |
| GR00T+SONIC | ST | `feat(GR00T-ST)/import-sub-task-feature-into-gr00t-sonic` | Existing branch; component evidence only |
| GR00T+SONIC | STR | `feat(GR00T-STR)/import-recovery-for-sub-task-into-gr00t-sonic` | Existing branch; component evidence only |
| GR00T+SONIC | ST-RT | `feat(GR00T-ST)/retrain-sub-task-vla-for-gr00t-sonic` | Existing user-chosen branch name; closed-loop result pending |
| GR00T+SONIC | STR-RT | `feat(GR00T-STR-RT)/retrain-recovery-vla-for-gr00t-sonic` | Existing branch; 40-D recovery retraining pending |
| PI0.5+SONIC | base | `feat(PI05)/reproduce-pi05-sonic-whole-body` | Existing PI0.5 deployment code/evidence, newly assigned architecture branch |
| PI0.5+SONIC | ST | `feat(PI05-ST)/import-sub-task-feature-into-pi05-sonic` | Language routing implemented; live integration pending |
| PI0.5+SONIC | STR | `feat(PI05-STR)/import-recovery-for-sub-task-into-pi05-sonic` | Recovery routing implemented; live integration pending |
| PI0.5+SONIC | ST-RT | `feat(PI05-ST-RT)/retrain-sub-task-vla-for-pi05-sonic` | Training launcher implemented; no selected checkpoint |
| PI0.5+SONIC | STR-RT | `feat(PI05-STR-RT)/retrain-recovery-vla-for-pi05-sonic` | Recovery-labelled training guard implemented; no selected checkpoint |

All ten named branches now exist. Retain the five GR00T branch names,
including the user-chosen ST-RT spelling. A branch remains partial until its
end-to-end method and checkpoint pass the execution ledger's gates.

## Benchmark branches (20)

For each architecture token in `GR00T`, `GR00T-ST`, `GR00T-STR`,
`GR00T-ST-RT`, `GR00T-STR-RT`, `PI05`, `PI05-ST`, `PI05-STR`,
`PI05-ST-RT`, and `PI05-STR-RT`, exactly these two branches now exist:

- `feat(<token>-benchmark-HA)/evaluate` — HumanoidArena.
- `feat(<token>-benchmark-SP)/evaluate` — SIMPLE.

Each branch currently contains the matching architecture ancestry, pinned
public source revision and fail-closed evaluation contract, but **not yet** a
completed v2 run. The branch must eventually contain its method's benchmark
integration, immutable run manifest, audit, metrics, representative video/plot
pointers, and result provenance. New commits advance that branch; do not
invent a new branch for each run, repair, or phase. Failed outcomes stay in
the record.

## Common evaluation contract

1. On HumanoidArena, run both families on the same G1/SONIC tasks, official
   success and fall predicates, horizons, initial states, evaluation modes,
   and paired rollout seeds. The current PI0.5 INT8 result is an external
   baseline, not automatically a matched ten-method comparison. Keep OpenDoor
   diagnostic until its pinned upstream contract mismatch is resolved.
2. On SIMPLE, first verify that both selected model checkpoints and every
   architecture adapter use the same G1/SONIC observation/action interface.
   The published GR00T N1.6 and OpenPi PI0.5 numbers are contextual references,
   not scores of this project's GR00T N1.7 or modified policies.
3. Report nominal task success and fall rate separately from failure-start and
   online-failure recovery. Use the same frozen failure states and budgets for
   all ten methods. A base model's failed trial is a scored outcome, not a
   reason to forbid another method's trial. The old nine-scenario 20/20 PI0.5
   gate is historical v0 evidence, not a gate for this v2 matrix.
4. Keep development, validation, and hidden-final splits disjoint. Choose
   checkpoints and the future `ours` source **only on validation** using a
   predeclared criterion; report the hidden-final result once without tuning
   on it. Match precision/backend or run a paired precision sensitivity audit
   before treating CPU and INT8 scores as interchangeable.
5. A branch is benchmark-complete only when its exact expected episode set,
   source/checkpoint hashes, per-task results, failure reasons, metrics, and
   visual evidence pass audit. Never infer completion from the branch name.

The future `ours` branch is conditional. It must be checked out from one of
the ten architecture branches only after a validation-based selection across
both benchmark suites. No `ours` branch exists yet; do not merge anything into
`dev` or `main` under this v2 plan before the selected contribution is ready.

Public benchmark sources: [HumanoidArena](https://github.com/William-wAng618/HumanoidArena),
[SIMPLE](https://github.com/physical-superintelligence-lab/SIMPLE), and
[LIBERO-RECOVER](https://arxiv.org/abs/2609.05178) for recovery taxonomy and
metrics, not its fixed-base robot scenes.
