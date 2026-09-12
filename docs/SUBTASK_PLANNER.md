# Closed-loop subtask planning for GR00T + SONIC

## Outcome and claim boundary

This repository adds a high-level planner that turns a long task into one bounded,
executable language subtask at a time. It is an **independent algorithmic
reproduction** of the selective test-time-compute design published by τ₀-VLA, with
NVIDIA Cosmos-Reason2-2B as the open proposal model and an auditable symbolic world
and value model for the included benchmark.

It is not an official τ₀-VLA implementation or checkpoint. As of the research cutoff
on 2026-09-12, the public τ₀-VLA repository exposed the low-level policy release but
not the high-level proposal, world, value, and reflection model weights. The exact
model revision, search configuration, and boundary of the claim are machine-readable
in [`config/subtask-planner.lock.json`](../config/subtask-planner.lock.json).

The included CPU results are symbolic planner evaluations. They are not physical
robot success rates. The Cosmos experiment measures next-subtask selection at
canonical decision points. SONIC regression measures the unchanged low-level
controller separately. A full end-to-end success claim still requires perception,
the task-specific GR00T post-training checkpoint, Isaac Sim task assets, and then
hardware safety validation.

## Research choice

The selected design combines the strongest compatible ideas found in the current
primary literature:

| Work | Relevant mechanism | Variant evaluated here | Reason for inclusion |
| --- | --- | --- | --- |
| [τ₀-VLA](https://arxiv.org/abs/2608.16885) | confidence routing, proposal/world/value decomposition, beam TTC, reflection | `best_of_n`, `ttc`, `adaptive_ttc` | Most complete current closed-loop hierarchy and strongest direct fit |
| [Vesta](https://arxiv.org/abs/2606.20905) | multimodal memory and explicit progress reasoning above GR00T/SONIC | `plan_once` plus observation-aligned memory | Confirms the same high/low-level separation on GR00T and SONIC |
| [Anticipation-VLA](https://arxiv.org/abs/2605.01772) | adaptive recursive subgoal generation | `recursive` | Tests look-ahead without alternative search |
| [STEP Planner](https://arxiv.org/abs/2506.21030) | hierarchical subgoal tree and leaf termination | `recursive` and bounded depth | Motivates explicit hierarchy and stopping conditions |
| [Hi Robot](https://proceedings.mlr.press/v267/shi25a.html) | language-commanded high-level policy over a low-level VLA | interface baseline | Establishes a simple hierarchical comparator |
| [Cosmos-Reason2](https://huggingface.co/nvidia/Cosmos-Reason2-2B) | open physical-AI reasoning VLM | proposal backend | Reproducible open substitute where τ₀ high-level weights are unavailable |

τ₀-VLA is the primary design because it does more than decompose once: uncertainty
selectively buys extra inference, candidate outcomes are compared before execution,
and the plan is reconsidered after each observation. This is a better operational
match for recoverable humanoid manipulation errors than an immutable list of steps.

## Architecture

At decision time the planner receives a large-task instruction, observation-derived
facts, an explicit skill catalog, and execution memory. It returns exactly one skill
ID. The adapter replaces only
`language["annotation.human.task_description"]` with that skill's natural-language
instruction before calling the existing GR00T policy.

```text
camera/state -> application-owned state extractor -> observed predicates
                                                     |
large task -> memory -> proposal model -> confidence router
                                  fast path ---------+----> affordance shield
                                  TTC path -> world model -> value -> beam search
                                                               |
                                      bounded subtask text -> GR00T (2.5 Hz)
                                                               |
                                      64-D latent + hands -> SONIC (50 Hz)
                                                               |
                                          new observation -> re-plan
```

The state extractor is deliberately not fabricated in this repository. Deployment
must supply perception whose predicates have defined detectors and thresholds. This
keeps predicted outcomes separate from observed outcomes.

## Algorithm

For a state `s`, proposal `a`, predicted outcome `s'`, skill cost `c`, and risk `r`,
the auditable benchmark value is:

```text
V(s,a,s') = 3.5 * (d(s)-d(s')) + 2.0 * goal_gain
            + 0.75 * milestone_gain - 0.08 * c - 0.7 * r
```

`d` is the shortest reachable symbolic distance to the task goal. A beam branch also
receives `0.15 * log(confidence)`. Search expands `branching_factor` proposals at
each level, retains `beam_width` branches, and stops at `search_depth` or when every
surviving branch reaches the goal.

The default `adaptive_ttc` route performs a single proposal first. It enters full
TTC if token-calibrated confidence is below `0.72` or the selected skill violates an
observed precondition. Otherwise it uses the fast path. Irrespective of route, the
final affordance shield rejects invented, inapplicable, or already-completed skills
and selects the best valid fallback.

Memory is reconciled from persistent observed milestones at every decision. It may
catch up, roll back, or repair itself; simulated predictions never become execution
facts. Every search branch owns a clone, preventing hypothetical outcomes from
leaking into sibling branches or live memory.

## Comparators

- `plan_once`: one direct next-subtask proposal, equivalent to a minimal high-level
  command policy.
- `recursive`: one proposed continuation per depth, testing anticipation without
  alternatives or value-guided branching.
- `best_of_n`: multiple immediate proposals ranked after one-step outcome prediction.
- `ttc`: full world-model-guided beam search at every decision.
- `adaptive_ttc`: fast direct route for confident valid proposals and TTC otherwise.

All variants share the same proposal distribution, task states, seeds, safety shield,
and metric code. Differences therefore isolate the search/routing algorithm in the
symbolic experiment.

## Benchmark protocol

[`benchmark/subtask_suite_v1.json`](../benchmark/subtask_suite_v1.json) contains six
long-horizon household tasks and 51 executable/recovery skills. Each task defines:

- initial and goal predicates;
- skill preconditions, effects, risk, cost, and stochastic execution probability;
- prerequisite-aware ordering through preconditions and priority;
- one deterministic, observable, recoverable failure;
- an explicit observation condition proving that the injected failure was recovered.

The CPU stress matrix uses identical seeds at proposal error rates `0.15`, `0.28`,
and `0.45`, covering nominal, standard, and adverse proposal quality. Each cell runs
all five methods across every task. `--workers 0` maps to every logical CPU. Raw
episode traces, exact arguments, Git revision, platform, chart tables, and rendered
charts remain under `results/subtask/`; raw traces use lossless `jsonl.gz`
compression and are not deleted after aggregation.

The GPU experiment loads the locked Cosmos-Reason2-2B snapshot in BF16, enables TF32
matrix operations, and evaluates the same canonical task states for every method.
Confidence is the geometric mean of the model-stated confidence and mean generated
token probability. Raw model text is retained for audit.

## Metrics

| Metric | Definition |
| --- | --- |
| Task success | Fraction of episodes whose full goal predicate set is observed by the step limit |
| Prerequisite-aware progress | Fraction of required skill milestones observed; recovery-only skills do not inflate it |
| Next-subtask accuracy | Selected skill equals the suite's deterministic valid priority label |
| Valid selection | Selected skill exists, satisfies current preconditions, and is not complete |
| Injected-failure recovery | Injected errors whose explicit observed recovery condition later becomes true |
| Safety fallback | Decisions for which the shield replaced an invalid proposal |
| Hallucination | Unknown proposed skill IDs per decision |
| Repetition | Consecutive identical selections per decision |
| TTC route | Decisions that spent full search compute |
| Compute | proposal calls, generated tokens, search nodes, mean latency, and p95 episode-mean latency |

Success and progress assess task completion; accuracy assesses agreement with one
canonical ordering. A different valid ordering can lower accuracy without lowering
success, so both must be reported. A failure counts as recovered only when its
declared observation predicate becomes true, never merely because some later action
succeeded.

## Reproduction

Run unit and integration-contract tests:

```bash
python3 -m unittest discover -s tests -v
```

Run one CPU cell with all logical cores:

```bash
python3 scripts/subtask_planner.py simulate \
  --output-dir results/subtask/cpu-noise-028 \
  --episodes-per-task 1000 --workers 0 --proposal-error-rate 0.28 \
  --branching-factor 4 --beam-width 3 --search-depth 3 \
  --confidence-threshold 0.72
```

Run the real proposal model from an immutable local snapshot:

```bash
_vendor/Isaac-GR00T/.venv/bin/python scripts/subtask_planner.py gpu-eval \
  --model-path /path/to/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561 \
  --output-dir results/subtask/cosmos-reason2 \
  --methods plan_once recursive best_of_n ttc adaptive_ttc \
  --max-points-per-task 2
```

The result manifest records the actual Git revision and hardware. Results must not be
interpreted from filenames alone; inspect each `manifest.json`, `summary.json`, raw
JSONL/JSONL.GZ trace, and chart CSV together.

## Deployment gates

The integration is ready to merge only after all of these gates pass:

1. suite validation and unit tests;
2. CPU stress matrix with bounded metrics and retained raw traces;
3. real Cosmos GPU inference with valid JSON recovery and retained raw output;
4. adapter contract proving only the GR00T language field changes;
5. existing SONIC release regression with unchanged action dimensions and stable run;
6. a clean diff and reviewed result/limitation report.

Physical deployment adds non-negotiable gates: perception calibration, collision and
joint-limit enforcement, operator stop, reduced-speed dry runs, Isaac Sim task-scene
validation, and task-specific GR00T/SONIC post-training. The symbolic affordance
shield is planning validation, not a replacement for a safety-rated robot controller.
