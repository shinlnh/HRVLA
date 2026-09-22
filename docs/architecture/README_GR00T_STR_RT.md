# GR00T-STR-RT architecture

This branch combines the GR00T 40-D semantic-action server, observed-state ST
router, BATON-inspired bounded recovery routing, and an in-domain GR00T
action-decoder retraining command. A live recovery request must carry a matched
failure label plus detector-produced current and pre-failure predicates; a
rollback decision emits **no** robot action until a verified checkpoint
mechanism is available. The GR00T/SONIC action contract is not modified.

Status: **PARTIAL**. The historical 43-DoF recovery-conditioned checkpoint is
retained only for provenance; its labels describe counterfactual recovery on
successful demonstrations. It is not a measured HumanoidArena fail-then-recover
checkpoint. The 40-D STR-RT launcher requires audited recovery-labelled data,
which are not present here. Real failure detectors, live Isaac Sim recovery,
selected checkpoint hashes, and paired benchmark evidence remain required.
