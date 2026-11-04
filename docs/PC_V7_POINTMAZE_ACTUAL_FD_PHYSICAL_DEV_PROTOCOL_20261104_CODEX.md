# v7 PointMaze actual finite-difference probes: exposed-root diagnostic

Fix this protocol before running. Reuse the two 21-action batches actually
scored for `first_response` at decision 2 of the sealed v7 root-0 run. Identify
the batches by the recorded nominal and exact requested ±0.05 coordinate
perturbations; verify their recorded response matrices. Do not generate,
select, or optimize actions using physical outcomes.

Replay the 15 warmup actions and the first two sealed dispatches, then each
recorded float32 five-step probe. For every branch reset the same shadow RGB
environment and check that the boundary frame equals the sealed tick-25
frame; save all five successor RGB frames. Derive terminal position and
velocity using the existing camera calibration and the fourth and fifth
successor frames. Separately replay identical actions on the native physics
worker; record all native states and compare its boundary to the existing
native action replay, not to an unrecorded shadow state.

Report the model, RGB-estimated and native physical 2x10 central-difference
matrices, absolute Frobenius error, relative error where denominator exceeds
1e-12 and cosine where both norms exceed 1e-12. Compare model score and physical
goal-distance pairwise ordering with ties excluded, plus the two endpoint
distance lists. RGB pixel quantization may dominate these small perturbations;
native and image-derived responses must remain separate. Model output and
physical positions need not share an offset, so compare derivatives, not
absolute coordinates. The native target is (1.8, 2.8); model scores are the
saved planner scores, not physical distances.

Preserve input hashes, actions, frames, native traces and analysis; do not
overwrite prior or partial outputs. Use the existing cooperative GPU lock,
DISPLAY=:0 and verify the expected physical GPU UUID before rendering.
No planner observes branch outcomes. This is one exposed root, one boundary,
two nominal actions and one perturbation size; it neither certifies PC1 nor
isolates model error from observation and backend differences.
