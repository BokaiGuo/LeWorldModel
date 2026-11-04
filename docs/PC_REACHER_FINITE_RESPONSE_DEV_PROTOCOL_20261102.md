# Reacher finite-response candidate screening: development protocol

Purpose: test whether a fixed, finite-action response signature produces a
different spectral-prefix shortlist before the existing predicted-successor
CEM selector. This is exploratory on the two already exposed Reacher roots,
not an independent test, P1 admission, or a physical claim.

Inputs are the existing guarded spectral candidate bank, root RGB, goal RGB,
executed action history, and frozen checkpoint. Candidate 0 (neutral) and
candidate 1 (unchecked DLS) are always retained. Each candidate's imagined
successor is probed with the same five one-packet suffix parameter vectors:
zero and ±0.3 along each of the two action axes, repeated at all five
positions of the ten-coordinate action token. For each candidate subtract
its own zero-suffix predicted planning coordinate from each nonzero-suffix
coordinate. This excludes the predicted baseline state displacement from the
signature, but does not establish task relevance or physical controllability.

Choose at most four additional original candidates by deterministic greedy
farthest-point sampling of Euclidean signature distance from the retained
set, breaking ties by original candidate index. Apply the unchanged
predicted-successor CEM selector to this shortlist with the original seed and
budget. Save original indices, signatures, full proposal parameters, source
hashes, model parameter hashes and separated planning-call counts. No
simulator state or post-execution outcome enters selection. Report no
selection when the existing selector finds no eligible candidate.

The probe cost includes one imagined prefix rollout per bank member and five
one-packet suffix rollouts per member, in addition to the guarded compiler
and CEM costs. Encoder and wall time are recorded separately where available,
and the report must not claim a fully certified equal-budget comparison.
This method-only development run does not test control; any physical
assessment must use sealed actions and fixed physical controls.
