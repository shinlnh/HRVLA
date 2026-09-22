# PI0.5-ST Isaac integration smoke (development only)

This folder copies the hash-bound `pp-box-smoke-v2` evidence from
`/HELIOS/Robotics/HRVLA/_artifacts/HumanoidArena/pi05-st-integration/`.
`audit.json` records the original paths and SHA-256 values; the copied episode,
trace, summary, frame, and video bytes match those hashes. Server and simulator
logs remain at the original artifact root and have hashes in `audit.json`.

The 120-step episode made six live PI0.5 HTTP requests through SONIC. All six
selected the first ST instruction and returned action40; no grasp transition
occurred before the short horizon. `timeout` is the observed behavior, not a
claim of task success. This development smoke does not satisfy the HA branch's
paired, full-horizon benchmark contract; `benchmark/v2_contract.json` remains
`pending`.
