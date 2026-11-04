# Fixed obstacle-chain prediction decomposition: development protocol

Use all 36 already exposed physical-oracle chains in their saved order: roots
8343, 8345, 8349, 8351 and bank indices 0–8. This is an offline diagnosis,
not action selection or independent evidence. Bind the existing oracle report
and arrays, model checkpoint, dataset sequence lengths, camera and scoring
inputs by SHA. Refuse an existing output and non-finite or malformed inputs.

Replay every fixed ten-action sequence in the RGB environment. Verify reset
and cp5 RGB against the source bank. Independently verify cp10 physical
position against the saved oracle position backend. For the same actions
record (1) open two-packet prediction from initial RGB, (2) one-packet
prediction after actual cp4/cp5 RGB velocity refresh, and (3) actual cp10 RGB
encoding with cp9/cp10 causal velocity. Use the same frozen checkpoint and
affine task score for all three, and separately record proprio latent MSE,
physical position distance, and the first-packet score. Predictions use only
RGB-derived inputs and recorded actions; simulator state is only for reset
and offline outcome verification. Physical outputs cannot select actions.

Report within-root pairwise ranking reversals and ties, plus per-candidate
values. Distinguish temporal RGB velocity averaging from true instantaneous
velocity. Explicitly account for model encodes, prediction samples, physical
steps and wall time. Check frozen parameter hashes. This tests the scoring
and fixed-action prediction chain only; it does not evaluate replanning,
independent generalization, strong baselines, or P1/PC admission.
