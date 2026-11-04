# Fixed-bank same-action cp5 refresh diagnostic

Development-only protocol, fixed before the run. Reuse all 384 sealed plans
in the eight adaptive CEM banks from roots 1884–1887, in saved order. Replay
each plan with the original legacy renderer, collecting both cp5 and cp10
RGB. Require exact cp10 image and endpoint equality with the sealed replay;
abort on mismatch. All plans execute the same ten recorded high-level actions,
irrespective of success. Do not select or exclude candidates using outcomes.

For each plan use the same frozen checkpoint, action normalizer, goal RGB,
and **initial** task mask to compare: (1) two-packet open-loop prediction
from initial RGB and RGB-derived position with zero velocity; (2) one-packet
prediction from its actual cp5 RGB and RGB-derived position with zero
velocity; (3) actual cp10 RGB encoding; and (4) actual terminal physical
distance for evaluation only. Recompute open-loop scores and require agreement
with the sealed source within rtol 1e-4 and atol 1e-6. Record masked visual
prediction MSE, pairwise rankings, choices and physical distances for each
bank, plus counts of encoding, prediction blocks, physical steps and wall
time. Recheck model parameters unchanged. Refuse overwriting results.

This is a paired, fixed-action model diagnostic. Obtaining a different true
cp5 for every candidate requires physically executing all candidate prefixes:
the resulting full-bank selector is an oracle, not an implementable
controller. Results do not establish reconfiguration, equal-budget advantage,
independent confirmation, or P1 admission. Roots and adaptive banks were
previously exposed; pairs within them are dependent. Unknown training-data
provenance remains unknown. Record failures without modifying this contract.
