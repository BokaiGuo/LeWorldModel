# Consistent optimization and replacement horizons

Prospective four-arm development ablation, fixed eligible return root0.
Compare against equal-six Q, not against a selectively chosen arm. Keep its
eight candidates, six updates, initial prefix protection, Q, seeds and evaluator.
No winner-only refinement. Run all four arms regardless of outcomes.

Identified code-level mismatch: feedback optimizer and its final replay used H3,
but the carry selector compared the new plan cropped to H2/H1. Thus gradients
could improve a future beyond the replacement comparison window.

Use H3/H2/H1/H1 at boundaries0/5/10/15 for Jacobian measurement, optimization,
candidate final replay and carry comparison. Keep three-packet action storage
for the existing ABI, but require the discarded tail to have zero gradient.
Immediate-progress screening remains H1. Carry re-evaluation remains separately
metered. Neural forward counts decrease with horizon; VJP counts stay unchanged.
Bind horizon into optimizer config and new source into optimizer identity.

This is a deliberate method variant, not retroactive invalidation of all prior
results. Does not establish that the mismatch caused a physical failure.
Report every arm, actual stopping step and cp0→cp5 physical component changes.
No native7-only non-greedy assertion, no training or gate changes.
Host CUDA only; no independent confirmation or equal-total-cost claim.
