# GR00T-STR architecture

This branch combines the GR00T state64/action40 server, observed-state
subtask planner, and BATON-inspired bounded recovery coordinator. The server
accepts a declared recovery protocol alongside the subtask program. It emits
a repair instruction only for a matched active failure with detector-produced
current and pre-failure predicates; a rollback decision emits no robot action.
Neither the VLA checkpoint nor SONIC's action contract is changed.

Status: **PARTIAL**. The protocol and server routing are unit-tested, while
real HumanoidArena failure detectors, verified rollback, and closed-loop
recovery traces are not yet attached to this architecture. Historical
symbolic and SONIC disturbance results are not a substitute for a full
GR00T+SONIC manipulation recovery rollout.
