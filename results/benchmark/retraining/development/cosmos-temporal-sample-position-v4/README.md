# Cosmos temporal sample-position v4 (negative development result)

This directory preserves the complete train/validation development result for
`cosmos-temporal-sample-position-v4`. No held-out episode was accessed. The
frozen confidence, agreement, minimum-phase, and residual-fallback gates were
not changed after observing the result.

- Candidates observed: 80/80
- Consensus accepted: 10 (train 9, validation 1)
- Residual fallback rate: train 0.122449, validation 0.142857
- Frozen maximum residual fallback rate: 0.05
- Admission result: **fail**
- Audit SHA-256: `2dcada7bbc17217073526a8148ade2052c0eae4ed74795cad856659f1ec0d297`

The dominant rejection reasons were undersized consensus phases (53 records)
and disagreement across the three staggered contact sheets (48 records). Four
records also exhausted bounded format repair. This negative result motivated
the pre-held-out v5 development protocol, which grounds each semantic
transition independently instead of asking the VLM to localize all boundaries
in one response.

Files in this directory retain the audit report, plot, every completed episode,
every raw model attempt, and one representative contact sheet for each
task/split combination.
