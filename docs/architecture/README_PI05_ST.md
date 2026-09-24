# PI0.5 + SONIC + ST

Status: **PARTIAL — one live Isaac integration smoke passed; no paired benchmark result**.

This branch reuses the PI0.5+SONIC task checkpoints and adds source-observed
sub-task instruction routing through the shared HumanoidArena language program.
The component test verifies that a grasp transition advances the selected
sub-task without enabling recovery or retraining. The bounded live test uses
the released task-matched PI0.5 checkpoint, the pinned CUDA-INT8 HTTP server,
SONIC and Isaac Sim. The integration runner guards the seven Gym-ID routes,
requires camera input, records each selected instruction, and rejects a
non-action40 policy chunk.

The first `pp_box` smoke attempt failed during Isaac's initial scene reset;
its log remains at `_artifacts/HumanoidArena/pi05-st-integration/pp-box-smoke-v1`.
The corrected `pp-box-smoke-v2` ran 120 control steps and made six real policy
requests using the first bounded ST instruction. The server log confirms it
received that instruction, every trace reports action40, and a video plus
decoded frame passed a hash-bound audit (`audit.json`, status
`integration_smoke_pass`). The episode timed out at the deliberately short
120-step limit; there was no observed grasp transition or task success. This
is integration evidence, not a HumanoidArena success-rate result.

Still required before claiming the HA benchmark: audit live prompt routes and
action40 transitions for the other six tasks, run full-horizon paired seeds
against the same PI0.5+SONIC base, and freeze checkpoint/runtime provenance.
The unit suite covers all seven task-ID mappings but does not replace live
simulator evidence. No ST success rate or SIMPLE adapter is claimed yet.
