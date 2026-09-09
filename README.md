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
python3 -m json.tool benchmark/suites/hrvla_recovery_v0.json >/dev/null
python3 -m json.tool benchmark/baselines/registry.json >/dev/null
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Score episode records

```bash
python3 scripts/score_results.py path/to/episodes.jsonl --output outputs/summary.json
```

See [`docs/BENCHMARK.md`](docs/BENCHMARK.md) for fairness rules, metrics, and the
boundary between direct reruns and paper-reported reference results.
