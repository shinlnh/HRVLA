# PI05-ST-RT architecture

Status: **PARTIAL — retraining command and method routing implemented; no trained checkpoint.**

Parent architecture: `feat(PI05-ST)/import-sub-task-feature-into-pi05-sonic`.

This branch provides a fail-closed LeRobot PI0.5 fine-tuning entry point and
registers `pi05_st_rt` with the existing ST instruction router. The command
requires a local PI0.5 base checkpoint and a 40-action/64-state LeRobot v3.0
dataset. The existing ST-RT dataset is v2.1; convert **a copy** using upstream
`convert_dataset_v21_to_v30.py` before training, since that converter replaces
its input directory. Keep original data and seed-specific outputs intact.

Still required: convert and audit the data, train and select a checkpoint,
bind its immutable hash to this manifest, verify it with the PI0.5 HTTP
provider and SONIC in Isaac Sim, then run a paired evaluation. `checkpoint:
null` deliberately prevents treating this branch as a finished RT baseline.
