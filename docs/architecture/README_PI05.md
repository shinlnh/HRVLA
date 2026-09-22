# PI0.5 + SONIC base architecture

Status: **CODE-READY** for the pinned HumanoidArena 64-D observation / 40-D
action interface. This is an architecture status, not a claim that the
ten-method paper comparison is complete.

The branch runs released task-specific PI0.5 checkpoints through the pinned
CUDA INT8 HTTP policy server and released SONIC encoder/decoder in Isaac Sim.
The seven task routes, official evaluator, and 84-cell/1,680-episode external
run are retained; the corrected PickPlaceBox route is selected transparently
while its 240 wrong-prompt originals remain historical evidence. The six-task
primary result excludes diagnostic OpenDoor, whose pinned task contract has a
known geometry/goal mismatch. No CPU/INT8 equivalence is claimed.

`benchmark/locks/pi05_ha/` pins all seven policy weight hashes and companion
metadata, both SONIC ONNX hashes, the simulator/source revisions, task horizons,
and the 1,440-identity primary seed plan. The fail-closed release auditor is
`scripts/audit_pi05_ha_baseline_release.py`; its completed result manifest,
metrics, and representative videos belong to
`feat(PI05-benchmark-HA)/evaluate`, not this architecture branch.

SIMPLE remains separate: its 43-D/78-D interface is incompatible with this
HumanoidArena policy path until a validated adapter or compatible checkpoint
is supplied. The SP benchmark branch is therefore still pending.
