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

A case may be `L3 + H3`; H3 is not a higher difficulty level than L4.

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
time, and paired McNemar tests when methods share episode keys.

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
