# Transition-aware subtask recovery for GR00T + SONIC

## Outcome and claim boundary

This repository adds a failure-recovery layer above the existing closed-loop
subtask planner. When a subtask fails, it chooses one bounded recovery primitive,
checks whether the primitive is executable from the observed state, and verifies
that its exit state is safe for the interrupted or successor subtask to inherit.
Only GR00T's language task description changes; the video, robot state, 64-D latent
action, 29-joint SONIC decoder, and 50 Hz control loop remain unchanged.

The implementation is an **independent algorithmic adaptation** of BATON, not the
authors' official code or checkpoint. The CPU benchmark is a symbolic fault-injection
experiment. The Cosmos experiment evaluates recovery selection at canonical failure
states. The Isaac Sim experiment separately regresses the unchanged SONIC whole-body
execution substrate. These results are not an end-to-end manipulation or physical
robot recovery success claim. The exact references, model revision, and claim
boundary are recorded in
[`config/subtask-recovery.lock.json`](../config/subtask-recovery.lock.json).

## Research choice

The literature search cutoff is 2026-09-13. BATON is the primary design because it
is the newest directly compatible work found and explicitly treats the transition
between subtasks as a first-class recovery problem.

| Work | Recovery mechanism | Use in this repository |
| --- | --- | --- |
| [BATON](https://arxiv.org/abs/2608.16889) | subtask-level exploration; invocation, handoff, and successor-lookahead transitions; transition memory | `baton_str`, the selected design |
| [ReSYNC](https://arxiv.org/abs/2606.18328) | learns recovery skills and relational planning concepts from failures | research comparator; not reproduced because it requires an RL training/domain pipeline |
| [AgentChord](https://arxiv.org/abs/2605.11951) | precompiles proactive recovery branches into an agentic task graph | mechanism-matched `agentchord` proxy |
| [Dream2Fix](https://arxiv.org/abs/2603.13528) | generates paired counterfactual failure/correction data and fine-tunes a recovery model | research comparator; not reproduced without its 120k-sample data engine |
| [ReKep](https://arxiv.org/abs/2409.01652) | replans/backtracks through relational keypoint constraints | controller/task-graph `rekep` proxy |
| [DoReMi](https://arxiv.org/abs/2307.00329) | detects plan/execution constraint violations and replans | frequent-monitor `doremi` proxy |
| [Inner Monologue](https://arxiv.org/abs/2207.05608) | incorporates environment feedback into language planning | reactive `inner_monologue` proxy |

The paper names in the benchmark denote isolated algorithmic mechanisms. They are
not official implementations, checkpoints, task assets, or reproductions of the
papers' reported experiments. All methods share the same task graph, fault states,
recovery catalog, random seeds, step budget, and metric code.

## Architecture

```text
observation predicates + failure detector
                    |
             readiness gate
                    |
       candidate recovery primitives
                    |
   transition memory + risk/cost score
                    |
    handoff contract + successor lookahead
                    |
 bounded recovery text -> GR00T -> SONIC
                    |
       observation-based verification
          | success             | failure
          v                     v
 resume interrupted edge   update memory / retry / checkpoint
```

Deployment owns the perception-to-predicate and failure-detection implementation.
Predicted effects never become observed facts automatically. A symbolic readiness or
handoff check is planning validation, not collision avoidance or a safety-rated
controller.

## BATON-STR decision rule

For a recovery primitive `r`, interrupted/successor subtask `j`, observed failure
state `s`, declared short-horizon success `p(r)`, empirical Beta posterior `B(r,j)`,
cost `c(r)`, and risk `q(r)`, the implementation ranks:

```text
E(r,j) = 0.55 * p(r) + 0.45 * B(r,j)

S(r,j) = 5.0 * full_handoff(r(s), j)
         + 3.0 * E(r,j)
         - 0.18 * c(r)
         - 0.90 * q(r)
         + 0.40 * explicit_failure_recovery(r(s))
```

Only readiness-valid candidates enter the ranking. The handoff term deliberately
dominates local cost: a quick correction is not useful if its residue makes the next
subtask fail. `B(r,j)` uses a Beta(1,1) prior keyed by failure, interrupted edge,
successor, and repair. If no forward repair is ready, the coordinator falls back to
the last verified task-graph checkpoint; this rewinds the controller, never
teleports the physical world.

The three transition checks are:

1. **Invocation:** call a recovery primitive only when its observed preconditions
   hold.
2. **Handoff:** require every declared entry predicate for the resumed edge.
3. **Lookahead:** key memory and scoring by the interrupted/successor edge, rather
   than judging local recovery in isolation.

## Benchmark protocol

[`benchmark/recovery_protocol_v1.json`](../benchmark/recovery_protocol_v1.json)
defines one observable failure for each of six long-horizon household tasks. Every
failure has three recovery alternatives:

- a cheap local repair that does not prepare the boundary;
- a precompiled forward branch that establishes part of the boundary;
- a transition-safe repair that establishes the complete handoff contract.

Every paired-seed episode injects the declared failure at least once and may inject
it again at the configured disturbance probability. An incomplete handoff adds a
task-specific boundary-failure risk proportional to the missing contract fraction.
The standard stress matrix runs six methods, six tasks, 3,000 episodes per task, and
disturbance rates 0.05, 0.15, and 0.35: 324,000 closed-loop episodes total. Raw
traces are retained losslessly as method-specific `jsonl.gz` files.

## Metrics

| Metric | Definition |
| --- | --- |
| Task success | fraction of episodes satisfying the full task goal before the action limit |
| Failure recovery | distinct injected failure incidents whose declared observed recovery predicate later holds |
| Handoff contract | mean fraction of required successor-entry predicates present at retry |
| Handoff failure | retry boundaries that fail according to missing-contract risk |
| Required progress | fraction of nominal required skill milestones observed; recovery actions do not inflate it |
| Recovery latency | executed actions from failure observation to verified recovery |
| Efficiency | total actions, recovery actions, monitor queries, model calls, and planner latency |
| Transition memory | fraction of recovery decisions with a previously observed matching edge |
| Escalation | episodes that exhaust applicable forward recovery and require fallback |

Task success and failure recovery include Wilson 95% confidence intervals. A
recovery is never credited merely because a later unrelated action succeeds.

## Reproduction

Validate the protocol and unit/integration contract:

```bash
python3 scripts/subtask_recovery.py validate
python3 -m unittest discover -s tests -v
```

Run one CPU cell:

```bash
python3 scripts/subtask_recovery.py simulate \
  --output-dir results/recovery/cpu-fault-015 \
  --episodes-per-task 3000 --workers 32 \
  --disturbance-rate 0.15 --proposal-error-rate 0.28
```

Aggregate multiple disturbance cells without deleting their raw traces:

```bash
python3 scripts/render_recovery_results.py \
  results/recovery/cpu-fault-005 \
  results/recovery/cpu-fault-015 \
  results/recovery/cpu-fault-035 \
  --output-dir results/recovery/comparison-final
```

Evaluate real Cosmos proposals from the immutable local snapshot:

```bash
_vendor/Isaac-GR00T/.venv/bin/python scripts/subtask_recovery.py gpu-eval \
  --model-path /path/to/Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561 \
  --output-dir results/recovery/cosmos-reason2-r20 \
  --candidates-per-failure 3 --repetitions 20
```

Each manifest records the exact Git revision, arguments, Python/platform, logical
CPU count, and GPU. Interpret results only together with the manifest, summary, raw
trace, chart data, and limitations report.

## Deployment gates

Merge is permitted only after protocol validation, all tests, the paired CPU stress
matrix, real Cosmos inference with retained raw generations, the unchanged-language
adapter contract, and fresh SONIC/Isaac Sim regression evidence all pass.

Physical deployment additionally requires calibrated perception and failure
detectors, task-specific manipulation scenes and GR00T post-training data, collision
and joint-limit enforcement, operator stop, reduced-speed dry runs, and hardware
validation. None of those safety gates is replaced by the symbolic benchmark.
