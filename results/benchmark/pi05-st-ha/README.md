# PI0.5 + SONIC + sub-task on HumanoidArena

Status: **running full matrix**, not a paper result. The baseline-only PI0.5 HA branch is
complete; this branch must independently evaluate the ST hook on its frozen
six-task primary identities. OpenDoor is diagnostic, never pooled into the
primary denominator.

The PI0.5 checkpoint, CUDA INT8 backend, SONIC controller, official task
predicates, task horizons, four modes, three group seeds, twenty repeats, and
episode-seed derivation are inherited from `benchmark/locks/pi05_ha/`.
The only intended method difference is the source-observed ST language
planner. The wrapper records one method trace and summary per episode.

Before the full matrix, run full-horizon seven-task pilot episodes and verify
that the batch runner produces atomic episode JSON, trace hashes, 40-D policy
actions, and at least one observed sub-task transition. A task failure or zero
transitions in an individual episode is an outcome, not permission to discard
or reroll it. This pilot is development evidence, not part of the 1,440 rows.
The pilot at architecture revision `573ef3da6ff80595e6f53cd2bcaa20ae1464968a`
completed 7/7 full-horizon episodes: 2 successes, 2 observed sub-task
transitions (SitSofa and OpenDoor), and 7/7 trace/summary/video integrity
checks. Five failures and all zero-transition rows remain recorded. The raw
pilot is at `_artifacts/HumanoidArena/pi05-st-pilot-v1/` on HELIOS.

Full-matrix command on the pinned workstation, after readiness passes:

```bash
cd /HELIOS/Robotics/HRVLA-worktrees/pi05-st-live
python3 scripts/run_humanoidarena_pi05_st_matrix.py \
  --runtime-root /HELIOS/Robotics/HRVLA \
  --output-root /HELIOS/Robotics/HRVLA/_artifacts/HumanoidArena/pi05-st-ha-cuda-int8-v1 \
  --tasks boxing doubledesk football pp_box sit_sofa vision_navi \
  --modes base_test semantic vision execution \
  --seeds 0 1 2 --repeats 20 \
  --policy-backend cuda_int8_weight_only \
  --record-video-every-n 10 --step-log-every-n 250
```

The same command resumes missing atomic episode JSON after an interruption.
The primary matrix started on HELIOS at `2026-09-22T07:04:32Z` as detached
driver PID `4021344`; the process log is
`_artifacts/HumanoidArena/pi05-st-ha-cuda-int8-v1/driver.log` and the live
progress file is the adjacent `progress.json`. At launch it reported 0/72
cells and 0/1,440 episodes. To check without touching the job:

```bash
python3 -m json.tool /HELIOS/Robotics/HRVLA/_artifacts/HumanoidArena/pi05-st-ha-cuda-int8-v1/progress.json
pgrep -af '^/usr/bin/python3 -u scripts/run_humanoidarena_pi05_st_matrix.py'
```

If interrupted, confirm the old driver and its child simulator/server are no
longer running, then rerun the exact full-matrix command above. Valid atomic
episode rows are skipped; never remove them to improve the outcome.

An unattended continuation was started at `2026-09-22T07:09:41Z` as PID
`4028209`. Its script is `scripts/continue_pi05_st_ha_after_primary.py` on
this branch. It waits for primary PID `4021344`, audits the primary 1,440
rows, runs the 240 OpenDoor diagnostic episodes separately, then audits those
rows. Its status is
`_artifacts/HumanoidArena/pi05-st-ha-release-v1/continuation-status.json`.
An `audits_passed_review_required` status means raw evidence passed automatic
checks; it does **not** silently mark this branch complete. Review the audit
manifests, generate/verify the paired comparison chart and video index, then
commit the final manifest and change `benchmark/v2_contract.json` to
`complete` only after a clean preflight.
Do not call this branch `complete` until all 1,440 primary rows, trace hashes,
sampled videos, paired identities, checkpoint/controller locks, confidence
intervals, plots, and the final result manifest have passed audit. Run the
240 OpenDoor episodes separately as a diagnostic with the same driver and
parameters except `--tasks open_door`.
