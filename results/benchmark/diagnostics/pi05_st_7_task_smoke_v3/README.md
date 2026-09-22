# PI0.5-ST: seven live HumanoidArena integration smokes

`matrix-audit.json` passed **7/7** bounded Isaac/SONIC/PI0.5 task routes at
architecture revision `e7842a9dbd1aa8df084a6149dab561c882a0c648`.
Each route used the same ST program, made three live HTTP requests with a
40-dimensional policy action, ran 60 control steps, and retained its raw
episode, instruction trace, server/simulator logs, screenshot, and video.
The audit re-hashes every retained file and verifies the first server prompt
against the selected ST instruction.

| Task | First instruction ID | Requests | Outcome at 60 steps |
| --- | --- | ---: | --- |
| `pp_box` | `approach_and_grasp_box` | 3 | timeout |
| `boxing` | `approach_strike_range` | 3 | timeout |
| `doubledesk` | `approach_and_grasp_hammer` | 3 | timeout |
| `football` | `approach_and_align_with_ball` | 3 | timeout |
| `sit_sofa` | `approach_sofa` | 3 | timeout |
| `vision_navi` | `leave_start_zone` | 3 | timeout |
| `open_door` | `approach_door_handle` | 3 | timeout |

Re-audit from this branch root:

```bash
python3 scripts/audit_pi05_st_live_matrix.py \
  --evidence-root results/benchmark/diagnostics/pi05_st_7_task_smoke_v3 \
  --output /tmp/pi05-st-matrix-audit.json
```

This is **integration evidence only**, not seven task successes or a paired
benchmark. The short horizon did not demonstrate any subtask transition.
The full HA branch still needs the registered full-horizon, paired-seed
base-versus-ST run and its precision/checkpoint/protocol locks; the v2 contract
therefore remains `pending`.

Earlier attempts are preserved under the remote artifact root
`/HELIOS/Robotics/HRVLA/_artifacts/HumanoidArena/pi05-st-integration/`.
`open_door-smoke-v1` made zero policy requests because the OpenDoor scene asset
is `XFormPrim`, not an articulation with `.data`; its traceback, episode, and
video are retained. `open_door-smoke-v2` verified the USD-handle fix. The
subsequent `v3` set above is the common-revision evidence and does not replace
or relabel either earlier attempt.
