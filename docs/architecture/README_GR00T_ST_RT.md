# GR00T-ST-RT architecture

This branch retains the original 43-DoF Arena action-decoder retraining
experiment for provenance and now adds a separate, guarded HumanoidArena
state64/action40 retraining launcher. It requires a selected shared GR00T
adaptation checkpoint, an ST-labelled LeRobot dataset, and the frozen
HumanoidArena modality configuration. The original 43-DoF checkpoint must not
be presented as a 40-D HumanoidArena RT checkpoint.

Status: **PARTIAL**. The in-domain training path is code-complete and rejects
wrong-dimensional inputs. Checkpoint selection, immutable artifact binding,
closed-loop Isaac Sim validation, and paired benchmark evidence remain
separate gates.
