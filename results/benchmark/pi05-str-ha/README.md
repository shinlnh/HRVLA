# PI0.5-STR HumanoidArena paired benchmark

This package freezes benchmark #3: PI0.5 + SONIC + sub-task routing + recovery-capable routing on the six-task HumanoidArena primary protocol.

The comparison uses the same 1,440 task × mode × group-seed × repeat identities as the locked PI0.5 baseline. Both methods use the same PI0.5 task checkpoints, SONIC controller, CUDA INT8 backend, simulator revisions, horizons, and camera/action40 interface. No failure is injected in this nominal matrix; recovery-injection performance is a separate claim.

## Audited result

| Task | PI0.5 baseline | PI0.5-STR | Delta |
|---|---:|---:|---:|
| Boxing | 163/240 (67.9%) | 22/240 (9.2%) | -58.8 pp |
| DoubleDesk | 96/240 (40.0%) | 82/240 (34.2%) | -5.8 pp |
| Football | 56/240 (23.3%) | 19/240 (7.9%) | -15.4 pp |
| PickPlaceBox | 145/240 (60.4%) | 135/240 (56.2%) | -4.2 pp |
| SitSofa | 131/240 (54.6%) | 123/240 (51.2%) | -3.3 pp |
| VisionNavigation | 47/240 (19.6%) | 0/240 (0.0%) | -19.6 pp |
| **Overall** | **638/1440 (44.3%)** | **381/1440 (26.5%)** | **-17.8 pp** |

PI0.5-STR Wilson 95% interval: 24.24%–28.80%. The paired task-stratified bootstrap 95% interval for STR minus baseline is -20.63 to -15.00 percentage points.

## Evidence

- `primary-audit.json`: all episode identities, hashes, method traces, action40 checks, success/failure outcomes, and video hashes.
- `comparison.json`: exact paired identities, discordant pairs, per-task metrics, and bootstrap interval.
- `comparison.svg`: report-ready visual comparison.
- `videos/`: one deterministic, outcome-independent sampled video per primary task.
- `result_manifest.json`: release-level hashes and headline metrics.

OpenDoor is excluded from the primary denominator by the frozen protocol because its upstream predicate remains diagnostic. This package makes no recovery-success, cross-family, or SIMPLE-suite claim.
