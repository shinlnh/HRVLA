# Qwen3-VL-4B constrained sequential grounding (model ablation)

This directory preserves the complete train/validation development comparison
for `Qwen/Qwen3-VL-4B-Instruct` at immutable revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`. It used the same v6 contact-sheet,
sequential constraint, confidence, phase-length, and all-three consensus
protocol as the Cosmos-Reason2-2B run. No held-out episode was accessed.

The fixed first-10 pilot favored Qwen by 7/10 versus Cosmos at 6/10, so the
comparison was completed rather than stopped early. The complete result did
not preserve that advantage:

- Candidates observed: 80/80
- Consensus accepted: 40 (train 35, validation 5)
- Residual fallback rate: train 0.069388, validation 0.085714
- Frozen maximum residual fallback rate: 0.05
- Admission result: **fail**
- Peak allocated VRAM: 8,762 MiB
- Audit SHA-256: `366abd652944f3410a0b47520b2061a609ec1b74bd301ecdc35e1d91dc895adf`

All 40 rejections were caused by disagreement across staggered views. Cosmos
v6 remained the stronger complete result (46 accepted; residual 0.059184 and
0.071429), so the larger Qwen model was not promoted. Raw generations,
completed episode records, representative contact sheets, and the plot are
retained here.
