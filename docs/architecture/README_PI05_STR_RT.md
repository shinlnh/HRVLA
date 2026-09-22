# PI05-STR-RT architecture

Status: **PARTIAL — retraining command and recovery routing implemented; no trained checkpoint.**

Parent architecture: `feat(PI05-STR)/import-recovery-for-sub-task-into-pi05-sonic`.

This branch combines the ST planner and bounded recovery instruction router
with a fail-closed PI0.5 fine-tuning entry point. Retraining must use
recovery-labelled demonstrations, not the ST-only dataset. The trainer
requires a 40-action/64-state LeRobot v3.0 dataset and a local PI0.5 base
checkpoint. Convert only a **copy** of a v2.1 dataset: the upstream converter
replaces its input directory.

Still required: collect/verify recovery-labelled trajectories, train and
select a checkpoint, bind its immutable hash, audit live PI0.5 HTTP + SONIC
recovery in Isaac Sim, and run paired evaluation. No claim-bearing benchmark
should use this branch until those checks pass.
