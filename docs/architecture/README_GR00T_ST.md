# GR00T-ST architecture

The ST branch combines the state64/action40 GR00T-to-SONIC bridge with
adaptive subtask planning. Launch the GR00T server with `--subtask-program`
to select a bounded skill instruction from detector-produced
`observed_predicates` in each `/infer` request. Missing or unknown predicates
fail closed; the planner never reads simulator ground truth or invents
observations. The VLA action and SONIC control contracts remain unchanged.

Status: **PARTIAL**. The HTTP path and fake-policy tests work, but the released
HumanoidArena client does not yet send those predicates. Implement and audit
scene-level detectors and the matching seven-task program before closed-loop
ST evaluation. This branch does not claim a trained ST checkpoint or a
successful simulation rollout.
