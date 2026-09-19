# Cosmos independent transition grounding v5 (negative smoke test)

This pre-held-out smoke test records the first development episode for the v5
protocol. Each of the two semantic transitions was queried independently over
three staggered contact sheets. All six model responses were schema-valid, but
each selected sample position S10. The resulting duplicate boundaries were
rejected without snapping or threshold changes.

- Candidates observed: 1/80 (development smoke test only)
- Schema-valid transition responses: 6/6
- Consensus accepted: 0
- Rejection: independently grounded boundaries were not strictly increasing
- Audit SHA-256: `9f31021b9462c2d5c8f8d6a63905c567cf3e7bf137f6685d2f12c9ad9a5d4942`

The result motivated v6 constrained sequential grounding, which exposes only
sample positions satisfying strict temporal order and the already-frozen
minimum phase duration. The v6 protocol does not alter a selected boundary
after inference and does not lower any admission threshold.
