# PI0.5 + SONIC + sub-task on HumanoidArena — frozen result

This branch contains the completed **PI0.5-ST HumanoidArena** evaluation. The
primary denominator is the same six tasks and the same 1,440
task/mode/seed/repeat identities used by the frozen PI0.5 baseline. OpenDoor's
240 episodes are retained as a separate diagnostic and are never pooled into
the primary score. No behavioral failure was discarded or rerolled.

| Primary task | PI0.5 baseline | PI0.5-ST | ST − baseline |
| --- | ---: | ---: | ---: |
| Boxing | 163/240 (67.92%) | 20/240 (8.33%) | -59.58 pp |
| DoubleDesk | 96/240 (40.00%) | 68/240 (28.33%) | -11.67 pp |
| Football | 56/240 (23.33%) | 21/240 (8.75%) | -14.58 pp |
| PickPlaceBox | 145/240 (60.42%) | 127/240 (52.92%) | -7.50 pp |
| SitSofa | 131/240 (54.58%) | 110/240 (45.83%) | -8.75 pp |
| VisionNavigation | 47/240 (19.58%) | 0/240 (0.00%) | -19.58 pp |
| **Six-task primary** | **638/1,440 (44.31%)** | **346/1,440 (24.03%)** | **-20.28 pp** |

The paired, task-stratified bootstrap 95% interval for the success-rate delta
is **[-23.06, -17.57] percentage points**. The ST run recorded 606 completed
sub-task transitions. Its Wilson 95% interval is 21.89–26.30%. The separately
labeled OpenDoor diagnostic recorded 2/240 successes (0.83%) and 35 completed
sub-task transitions.

The negative result is retained unchanged: this ST implementation is worse
than the matched PI0.5 baseline under the frozen nominal protocol. It is not
evidence against every sub-task planner, and it is not a recovery or
cross-family result.

Evidence in this directory:

- `primary-audit.json`: all 1,440 primary episode identities, raw hashes,
  method-trace hashes, checkpoint/controller/source checks, metrics and CIs.
- `diagnostic-audit.json`: the separate 240-row OpenDoor audit.
- `comparison.json`: exact paired outcomes and per-task summaries.
- `comparison.svg`: the checked-in metric chart.
- `videos/`: deterministic, non-outcome-selected samples for six primary tasks
  plus OpenDoor diagnostic.
- `result_manifest.json`: compact release boundary and SHA-256 bindings.

Recheck on the pinned workstation:

```bash
python3 scripts/benchmark_v2_preflight.py \
  --suite-root /HELIOS/Robotics/HRVLA/_vendor/HumanoidArena
python3 scripts/audit_pi05_st_ha_release.py
```

The PI0.5-ST architecture revision, CUDA INT8 backend, task checkpoints, SONIC
controller, simulator revisions, horizons and seed plan are frozen by
`benchmark/v2_contract.json` and `benchmark/locks/pi05_ha/`. Sample videos are
visual evidence only, not statistically representative outcomes.
