# Cosmos constrained sequential grounding v6 (negative development result)

This directory preserves the complete train/validation development result for
`cosmos-constrained-sequential-grounding-v6`. No held-out episode was accessed.
Each transition was grounded sequentially, and the decoder exposed only sample
positions satisfying strict order and the already-frozen minimum phase length.
No selected boundary was snapped or changed after model inference.

- Candidates observed: 80/80
- Consensus accepted: 46 (train 40, validation 6)
- Residual fallback rate: train 0.059184, validation 0.071429
- Frozen maximum residual fallback rate: 0.05
- Admission result: **fail**
- Audit SHA-256: `2f3dba741b2b0866d6f21b5051d7f7020eea4690c147e281388d9be04f90e3ea`

Compared with the v4 joint-grounding result (train 0.122449, validation
0.142857), v6 substantially reduced residual fallback. The remaining dominant
failure was disagreement across the three staggered contact sheets (32
records); two records exhausted bounded format repair. Because v6 still missed
the frozen gate, it was retained as a negative ablation rather than promoted.

Files in this directory retain the audit report, plot, all completed episodes,
all raw model attempts, and representative contact sheets.
