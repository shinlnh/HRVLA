# PI0.5 + SONIC on HumanoidArena — frozen baseline

This branch contains one **completed baseline evaluation**, not a completed
ten-method paper comparison. The primary denominator is six compatible tasks:
4 modes × 3 group seeds × 20 repeats × 6 tasks = **1,440 episodes**.
OpenDoor's 240 episodes remain a separate diagnostic because the pinned task
contract has an unresolved upstream geometry/goal mismatch. No outcomes were
discarded or rerolled to improve the score.

| Primary task | Successes / 240 | Success rate |
| --- | ---: | ---: |
| Boxing | 163 | 67.92% |
| DoubleDesk | 96 | 40.00% |
| Football | 56 | 23.33% |
| PickPlaceBox, corrected prompt route | 145 | 60.42% |
| SitSofa | 131 | 54.58% |
| VisionNavigation | 47 | 19.58% |
| **Six-task primary** | **638 / 1,440** | **44.31%** |

OpenDoor diagnostic: 88/240 (36.67%). Across all seven recorded tasks the
descriptive total is 726/1,680 (43.21%), but that total is **not** the
six-task primary estimand. The six-task Wilson 95% interval is 41.76–46.88%.

The original 1,680-episode run had the wrong PI0.5 language route only for
PickPlaceBox. Its 240 original records remain intact and excluded. A separately
labeled same-seed, same-checkpoint, same-backend rerun supplied the corrected
240 rows; the other six task rows came unchanged from the original run. The
[amended source manifest](../external/pi05_cuda_int8_v1_amended_summary.json)
records all 1,680 selected episode hashes and 168 selected video hashes.

The [release audit](result_manifest.json) recompiled every raw episode and
video from `/HELIOS/Robotics/HRVLA/_artifacts/HumanoidArena/paper-baselines/`,
verified all seven 9.35 GB policy weight files and metadata, the SONIC ONNX
files, source revisions, corrected prompt log, locked seed identities, and
the six-task denominator. The four immutable input locks are in
`benchmark/locks/pi05_ha/`. The [visual audit](visual_audit.json) binds the
seven checked-in [sample videos](videos/) byte-for-byte to the frozen source
manifest; the selection rule is lexicographic first per task, not a
statistical representation of outcomes. The [metric chart](../external/pi05_cuda_int8_v1_amended_summary.png)
is retained with the raw metrics.

Recheck on the pinned workstation:

```bash
python3 scripts/benchmark_v2_preflight.py \
  --suite-root /HELIOS/Robotics/HRVLA/_vendor/HumanoidArena
python3 scripts/audit_pi05_ha_visuals.py \
  --output /tmp/pi05-ha-visual-recheck.json
python3 scripts/audit_pi05_ha_baseline_release.py \
  --runtime-root /HELIOS/Robotics/HRVLA \
  --output /tmp/pi05-ha-release-recheck.json
```

The lock was made explicit after the original baseline run; it is a
retrospectively frozen baseline, not a preregistered prospective experiment.
Future architectures must run the same primary task/mode/seed/repeat identities,
controller, backend or a separately audited precision bridge, and official
predicates before a paired comparison is claimed. This result does **not**
prove CPU/INT8 equivalence or any advantage over another method.
