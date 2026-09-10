# HRVLA recovery benchmark

This branch defines the evaluation contract for comparing HRVLA with official
baselines. It intentionally contains no pre-filled paper results and makes no
claim that unlike tasks are directly comparable.

The benchmark separates three questions:

1. `nominal`: does adding recovery preserve normal task execution?
2. `failure_start`: can a method recover when every method starts from the same
   post-failure simulator snapshot?
3. `online_failure`: can the complete system detect, diagnose, recover, and
   continue after the same event-triggered failure injection?

Recovery difficulty follows LIBERO-RECOVER's L1-L4 taxonomy. Humanoid-specific
body/tracker failures are an orthogonal H1-H3 axis, not invented L5-L7 levels.

## Validate the specification

```bash
python3 -m json.tool benchmark/spec/episode.schema.json >/dev/null
python3 -m json.tool benchmark/spec/plan.schema.json >/dev/null
python3 -m json.tool benchmark/spec/run-manifest.schema.json >/dev/null
python3 -m json.tool benchmark/suites/hrvla_recovery_v0.json >/dev/null
python3 -m json.tool benchmark/baselines/registry.json >/dev/null
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Generate a deterministic paired plan and provenance manifest (draft scenarios
are excluded unless explicitly requested):

```bash
python3 scripts/benchmark.py plan benchmark/suites/hrvla_recovery_v0.json \
  --method gear_sonic_original_release --method hrvla \
  --output outputs/plan.json --manifest outputs/run-manifest.json
```

## Score episode records

```bash
python3 scripts/benchmark.py score path/to/episodes.jsonl \
  --output outputs/summary.json
```

See [`docs/BENCHMARK.md`](docs/BENCHMARK.md) for fairness rules, metrics, and the
boundary between direct reruns and paper-reported reference results.

## Validated SONIC backend

This branch includes the locked GEAR-SONIC reproduction used as the benchmark
execution backend. On the RTX 5070 Ti workstation, Isaac Sim 5.1.0 and Isaac
Lab 2.3.2 completed both bundled walk-forward motions (`4,004` frames) without
termination. The [machine-readable result](results/sonic-default-sample.json),
[runtime instructions](docs/SONIC_BASELINE.md), and source/artifact lock files
are part of the benchmark provenance.

The executed controller-track artifacts are committed under
[`results/benchmark/`](results/benchmark/): nominal SONIC motion tracking and an
audited one-shot H1 lateral-push diagnostic. Raw simulator metrics are retained
beside their scored reports so every recorded SHA-256 can be checked locally.

![GEAR-SONIC controlling Unitree G1 in Isaac Sim 5.1](docs/assets/sonic-isaac-sim-5.1.png)
