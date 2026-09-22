# PI05-ST on HumanoidArena

Status: **RUNNING — not yet a benchmark result.**

Architecture parent: `feat(PI05-ST)/import-sub-task-feature-into-pi05-sonic`.
Public benchmark: HumanoidArena.

The architecture has passed seven bounded smokes and a separate seven-task
full-horizon pilot, including observed sub-task transitions on SitSofa and
OpenDoor. The locked CUDA INT8 primary matrix is now running on the same
1,440 six-task episode identities as the frozen PI0.5 baseline. Its exact
command, PID, progress path, and unattended diagnostic/audit continuation are
in [`results/benchmark/pi05-st-ha/README.md`](../../results/benchmark/pi05-st-ha/README.md).
The PI0.5 HumanoidArena INT8 baseline is historical evidence until the ST
matrix and exact paired comparison pass audit. OpenDoor remains a separate
240-episode diagnostic, not part of the primary denominator.

Do not interpret this README, branch name, pilot, or process launch as a
completed run. `benchmark/v2_contract.json` remains `ready`, not `complete`.
