# Transparent humanoid recovery evaluation

## Claim boundary

HRVLA asks whether a humanoid VLA can detect a failure, restore the body/object/
environment state, and resume the original long-horizon task. Most related
papers optimize a different primary question. Therefore:

- paper-reported numbers are contextual references, not rows in one ranking;
- a direct “better than” claim is allowed only after both methods run on the
  same HRVLA episode records and budgets;
- if official code/checkpoints are unavailable, report “not rerun” instead of
  implementing an approximation under the paper's name;
- component-level claims must name the slice, such as object-drop L3 recovery,
  rather than imply universal superiority.

The baseline registry records this boundary per work.

## Evaluation tracks

Never pool results from different tracks:

- `controller_tracking_pilot` verifies that the locked whole-body checkpoint can
  execute its official motion inputs in physics;
- `end_to_end_recovery` evaluates task success under nominal, failure-start,
  and online-failure protocols;
- paper-reported values remain contextual metadata and are never converted into
  synthetic episode records.

The first tracked pilot uses the exact SONIC checkpoint from the baseline
branch. Both official walk-forward motions completed (`2/2`, 4,004 frames,
80.04 s); local MPJPE was `23.55 mm`. This proves the controller path is alive,
not that a G1 can grasp, recover, or outperform another paper. See
[`results/benchmark/sonic-official-sample-v0.json`](../results/benchmark/sonic-official-sample-v0.json).

An additional H1 diagnostic injected a one-shot `+0.35 m/s` world-frame
lateral root velocity at simulator step 101 (about 2.0 s) into both official
motions. The injector wrote an audit record before scoring; both motions still
completed (`2/2`, zero terminations). Mean local MPJPE changed from `23.55 mm`
to `17.56 mm`. Because the perturbation happened to improve this tiny sample,
the result is reported as a beneficial perturbation, not as evidence of
recovery. The timer is only a gait-settle proxy and the scenario lacks the
required independent 20/20 oracle admission. See
[`results/benchmark/sonic-h1-push-pilot-v0.json`](../results/benchmark/sonic-h1-push-pilot-v0.json).

## Protocols

### Nominal

Run without injected failure. Report task success, fall rate, and time/budget.
The recovery-enabled system must not gain an unfair retry or horizon extension.

### Failure-start

Serialize a simulator state after a real or injected failure and give the exact
snapshot to every method. The pre-failure trajectory is excluded. This is the
cleanest measure of recovery competence.

### Online failure

Start from the shared nominal initial state and inject at a semantic event such
as `after-stable-grasp`, `after-door-crossing`, or `before-object-placement`.
The event detector, perturbation parameters, and state transition are shared.
This measures the complete detect-to-resume system.

Wall-clock timestamps are not sufficient triggers: different policies reach a
different semantic state at the same time.

## Failure labels

L1-L4 are inherited from LIBERO-RECOVER:

- L1: action retry;
- L2: action adaptation;
- L3: task-object state restoration;
- L4: blocking environment state restoration.

Humanoid consequences form a separate multi-label axis:

- H1: support/balance/foot-placement/body-state recovery;
- H2: valid high-level intent but failed whole-body tracker execution;
- H3: coupled body and object/task-state disruption.

A case may be `L3 + H3`, or `H1` alone for a pure body disturbance; H3 is not
a higher difficulty level than L4.

## Scenario admission

Every recovery scenario must include a deterministic snapshot or deterministic
event-triggered injection and an independent recoverability check. Reject states
that an expert/oracle cannot recover within the common observation and action
budget. Preserve rejected scenarios and the rejection reason outside the scored
test set.

Freeze before evaluation:

- task success and failure predicates;
- fall and safety termination predicates;
- snapshots, event boundaries, perturbation parameters, and seeds;
- observation fields, control rate, controller revision, and horizon;
- allowed compute, model checkpoint, recovery memory, and reset policy.

The seven task-success predicates must resolve to the exact reward source used
by the locked evaluator. Safety/fall rates remain separate outcomes; they must
not be silently conjoined with or removed from the upstream binary task reward.
Run `scripts/audit_humanoidarena_recovery_contract.py` to hash those source
files and expose the still-missing snapshot, runtime-injector, and oracle gates.

Initial and failure-start states use
`src/hrvla_bench/isaac_snapshot.py`. A snapshot contains the exact 13-value root
state and, for articulations, ordered joint position/velocity values for every
declared scene asset. Task/event/seed/step and simulator revision are inside the
canonical SHA-256 envelope. Restore rejects hash or joint-order drift and uses
Isaac Lab's public root/joint state writers. Wall-clock timestamps are excluded,
so an identical simulator state has an identical content address.

Action-stream injectors are typed by interface. `release-grasp-contact` may
only force the two semantic-action40 hand bits open, while
`attenuate-sonic-latent` may only scale the post-encoder latent64; applying
either at the wrong dimensional seam is a hard error. World-frame root-velocity
events record before/after values and use per-injector, per-environment flags.
The evaluator must call `reset_one_shot_injectors` at every episode reset so a
persistent simulator cannot silently skip perturbations in later rollouts.
Scene-level primitives resolve lateral deltas in the live asset frame, retain
before/after poses and velocities, and represent a body impulse as force over
exactly one controller interval before clearing its Isaac Lab wrench buffer.
The support-foot event selects the ankle-roll body with the larger live contact
force; it does not kinematically teleport an articulated foot.

Every recovery event also carries a typed detector contract. Boundaries must
be observable before the task success predicate fires. In particular,
DoubleDesk uses hammer transport rather than a nonexistent second transfer,
Boxing uses the pre-hit approach shell rather than the terminal hit, and visual
navigation uses displacement from the immutable start rather than a nonexistent
waypoint. OpenDoor locks `OPEN_DOOR_STRICT_REQUIRE_GEOMETRY=1`; otherwise a
post-open obstruction snapshot would already satisfy the default strict reward.

`scripts/run_humanoidarena_recovery_episode.py` wraps the immutable upstream
evaluator without editing its checkout. It accepts exactly one episode, hooks
the action40 or post-encoder latent64 seam selected by the suite, evaluates the
source-backed detector every control step, and embeds a recovery summary in the
ordinary episode JSON. Scene events and one-control-step forces are written to
`injector-audit.jsonl`; detector edges and hash-linked action changes go to
`runtime-trace.jsonl`. Optional
initial/failure capture writes hash-addressed snapshots and refuses to replace a
different state at the same path. Failure-start capture uses the declared settle
interval and must run before any recovery instruction is supplied.

The wrapper takes all normal upstream evaluator arguments after its recovery
arguments. Run it with the locked Isaac Lab Python and the same SONIC/server
flags used by the matrix runner, for example:

```bash
_vendor/IsaacLab-v2.2.0/.venv/bin/python \
  scripts/run_humanoidarena_recovery_episode.py \
  --recovery-scenario support-state-push \
  --recovery-output-dir _artifacts/HumanoidArena/recovery-admission/support-state-push \
  --task Isaac-Move-Football-Single-G129-Dex3-Wholebody \
  --seed 0 --episode_seed 20260915 \
  ...the locked upstream SONIC and HTTP-server arguments...
```

This command producing a trace is not itself admission. A trace auditor must
still prove the detector edge, exactly one perturbation, compatible snapshots,
and independent oracle success before the suite status can change from draft.
The fail-closed runtime portion is checked separately:

```bash
python3 scripts/audit_humanoidarena_recovery_runtime.py \
  --scenario support-state-push \
  --runtime-dir _artifacts/HumanoidArena/recovery-admission/support-state-push \
  --result-json /path/to/the/single/episode.json \
  --output /path/to/runtime-audit.json
```

It verifies the suite hash, seed, simulator revision, detector edge, interface
width, numerical before/after action hashes, physical magnitude and target,
one-shot timing, and the exact failure-snapshot settle boundary. Passing this
audit means only that the runtime injector evidence is valid; oracle admission
and recoverability remain pending.

All end-to-end scenarios in `hrvla_recovery_v0` intentionally start as
`draft`. The suite now uses the same HumanoidArena state64/semantic-action40
contract as the controlled internal matrix; its older latent64/Isaac Sim 5.1
placeholder contract was retired before any claim-bearing plan was generated.
A draft becomes `admitted` only after its immutable artifacts are published,
its predicates are tested, and an independent oracle succeeds in all 20
admission trials. The default planner excludes drafts, so a missing dataset
cannot silently become a publishable score.

## Reproducible pipeline

Validate the suite and generate an immutable paired plan:

```bash
PYTHONPATH=src python3 -m hrvla_bench.cli validate-suite \
  benchmark/suites/hrvla_recovery_v0.json
PYTHONPATH=src python3 -m hrvla_bench.cli plan \
  benchmark/suites/hrvla_recovery_v0.json \
  --method gear_sonic_original_release \
  --method hrvla \
  --output outputs/plan.json \
  --manifest outputs/run-manifest.json
```

The suite SHA-256 and plan SHA-256 freeze task cells independently of method
order. Each cell fixes the initial/failure snapshots, semantic event,
injector parameters, training seed, rollout seed, controller, horizon, and
success predicate.

Method integrations implement `run_episode(episode)` and are loaded through a
`package.module:attribute` adapter. The runner protects planned identifiers,
refuses non-admitted states by default, validates every returned episode, and
fsyncs each episode to a validated `.partial` JSONL. A retry resumes at the
first missing plan entry; only a complete run atomically replaces the final
JSONL:

```bash
PYTHONPATH=src python3 -m hrvla_bench.cli run outputs/plan.json \
  --adapter my_method.hrvla_adapter:create_adapter \
  --method hrvla \
  --track end_to_end_recovery \
  --run-id hrvla-v0-s0 \
  --checkpoint-id sha256:... \
  --controller-id gear-sonic:e6bdab3... \
  --simulator-revision isaac-sim-5.1+isaaclab-37ddf62 \
  --output outputs/hrvla-v0-s0.jsonl
```

The `--allow-draft` option is for harness development only. Its outputs are not
eligible for claim-bearing reports.

## Claim proof gates

A syntactically valid suite is not necessarily ready to support a scientific
claim. Before generating the final plan, run:

```bash
PYTHONPATH=src python3 -m hrvla_bench.cli validate-methods \
  benchmark/methods/registry.json \
  --artifact-lock config/benchmark-artifacts.lock.json
```

The method gate fixes the meaning of ST, STR, and RT, resolves every
implementation to a Git commit, and requires retrained checkpoints to match the
published artifact lock. The registered ablations change one named component at
a time; paper proxies are not registered as official runnable methods.

```bash
PYTHONPATH=src python3 -m hrvla_bench.cli claim-readiness \
  benchmark/suites/hrvla_recovery_v0.json \
  --output outputs/claim-readiness.json
```

The command exits with status 3 until every included task and recovery scenario
is admitted. Admission requires immutable snapshot hashes, tested task/failure
predicates, a versioned injector, and at least 20 successful oracle trials out of
20. Failure-start scenarios additionally require a failure-snapshot hash. This
gate is expected to fail for the current v0 draft and prevents a development
result from silently becoming claim-bearing evidence.

After running all methods, audit the separate JSONL files against the same frozen
plan:

```bash
PYTHONPATH=src python3 -m hrvla_bench.cli audit-evidence outputs/plan.json \
  outputs/baseline.jsonl outputs/candidate.jsonl \
  --output outputs/evidence-audit.json
```

This second gate rejects missing or duplicate method/episode cells, unplanned
records, draft recovery states, suite/plan hash mismatches, checkpoint changes
within a method/training-seed cell, and controller or simulator changes across
paired methods. Its hash-addressed report is the prerequisite for a comparative
score or table.

Validate and score without installing extra dependencies:

```bash
PYTHONPATH=src python3 -m hrvla_bench.cli validate-records outputs/episodes.jsonl
PYTHONPATH=src python3 -m hrvla_bench.cli score outputs/episodes.jsonl \
  --output outputs/summary.json
```

Disturbance smoke tests remain outside the claim-bearing JSONL. Run the locked
Sonic checkpoint with the audited one-shot injector:

```bash
export HRVLA_ISAACSIM_ROOT=/home/shin/isaacsim
export HRVLA_ISAACLAB_ROOT=/home/shin/IsaacLab
python3 scripts/run_sonic_h1_pilot.py --runtime workstation
```

The wrapper reuses GEAR-SONIC's typed `push_robot` event slot, schedules it at
step 101 (about 2.0 s), replaces the event function with HRVLA's one-shot
wrapper, and writes `injection-audit.jsonl`. Rebuild the tracked report from the
committed raw metrics and audit with:

```bash
PYTHONPATH=src python3 -m hrvla_bench.cli report-sonic-disturbance \
  results/benchmark/raw/sonic-nominal-metrics.json \
  results/benchmark/raw/sonic-h1-push-metrics.json \
  results/benchmark/raw/sonic-h1-push-audit.jsonl \
  --run-id sonic-h1-push-pilot-20260910 \
  --checkpoint-id 'hf://nvidia/GEAR-SONIC@<revision>/sonic_release/last.pt#sha256=<sha256>' \
  --controller-id 'gear-sonic@<revision>' \
  --simulator-revision 'isaac-sim-5.1.0+isaaclab@<revision>' \
  --recorded-at '2026-09-10T07:04:26+00:00' \
  --output results/benchmark/sonic-h1-push-pilot-v0.json
```

The committed report links each raw input path and its SHA-256. The raw files
are small enough to review in Git; simulator caches and model artifacts remain
ignored.

## Reporting

Use three training seeds and 20 rollout trials per seed per comparison cell as
the initial protocol, following HumanoidArena. Report all seeds, not only their
mean. Increase repeats after a power analysis if expected differences are
small.

Primary tables must include nominal SR, failure-start macro RSR, online macro
RSR, RSR by L-level and H-axis, fall rate, and safety-violation rate. Macro RSR
first averages within `(task, L, H-label-set, severity)` cells so a frequent easy
failure cannot dominate the ranking. Micro RSR and Wilson intervals remain
diagnostics.

The scorer also reports the LIBERO-RECOVER-inspired recovery degradation (RD)
and recovery consistency (RC), detection precision/recall/F1, latency, recovery
time, and paired McNemar tests when methods share episode keys. Pairing also
requires identical suite, snapshots, controller, simulator revision, and seeds;
Holm-adjusted p-values are emitted across the comparison family.

Do not combine `nominal`, `failure_start`, and `online_failure` into one headline
score; they answer different questions.

## Paper-specific use

- HumanoidArena supplies leg-critical task coverage, common initialization,
  fall/execution reporting, and the 3 x 20 repeat convention.
- LIBERO-RECOVER supplies the recovery-state definition, L1-L4 taxonomy, RSR,
  RD, and RC, but is fixed-base manipulation.
- Pro-HOI is relevant to L3 object-drop detection, pose recovery, and regrasp.
- OmniContact is relevant to long-horizon replanning after dropped or displaced
  objects.
- SUGAR is relevant to disturbance robustness and autonomous recovery, while
  its main contribution is scalable human-video skill learning.
- HumanoidExo is primarily a data-collection and whole-body manipulation
  baseline, not a dedicated recovery benchmark.

These become directly comparable only through a matched rerun or an explicitly
matched subset protocol.

## Primary sources

- [LIBERO-RECOVER](https://arxiv.org/abs/2609.05178)
- [HumanoidArena](https://arxiv.org/abs/2606.17833)
- [Pro-HOI](https://arxiv.org/abs/2603.01126)
- [OmniContact](https://arxiv.org/abs/2606.26201)
- [SUGAR](https://arxiv.org/abs/2605.20373)
- [HumanoidExo](https://arxiv.org/abs/2510.03022)
