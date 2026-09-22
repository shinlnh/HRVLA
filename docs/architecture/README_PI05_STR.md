# PI05-STR architecture

Status: **PARTIAL — recovery instruction routing implemented, live integration pending.**

Parent architecture: `feat(PI05-ST)/import-sub-task-feature-into-pi05-sonic`.

This branch adds a PI0.5 method identity that uses the same ST planner and
switches to a bounded recovery instruction only while a declared failure is
active. It does not retrain or replace the PI0.5 checkpoint.

Still required: connect and audit all nine failure injectors with the PI0.5
HTTP provider in Isaac Sim, validate recovery trajectories and snapshots, and
freeze runtime/checkpoint hashes. Do not launch a claim-bearing benchmark yet.
