# PI0.5 + SONIC + ST

Status: **PARTIAL — architecture branch exists; no closed-loop result**.

This branch reuses the PI0.5+SONIC task checkpoints and adds source-observed
sub-task instruction routing through the shared HumanoidArena language program.
The component test verifies that a grasp transition advances the selected
sub-task without enabling recovery or retraining.

Still required before benchmarking: validate the PI0.5 HTTP-provider hook in
Isaac Sim, audit task-specific prompt routes and action40 transitions for all
seven tasks, then freeze checkpoint/runtime provenance. No ST success rate or
SIMPLE adapter is claimed by this branch.
