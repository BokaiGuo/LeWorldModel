# Two-packet same-action prediction and physical diagnostic

On all four exposed roots execute all nine sealed spectral-bank prefixes
followed by their own saved optimized five-action suffix. Convert each
saved normalized suffix to raw actions once, clip to the valid raw box,
then normalize the exact float32 actions sent to physics for model replay.
Record the maximum normalized round-trip discrepancy and assert the
recomputed model cost matches the sealed selected cost within `rtol=2e-5,
atol=2e-7`. Verify step-5 RGB against the sealed prefix bank.

At step 10 compare three same-action proprio-token quantities: open-loop
two-packet prediction from initial RGB, one-packet prediction from actual
step-5 RGB and causal frame-difference velocity, and actual step-10 RGB
encoding with causal velocity. Report their target-token distances and
pairwise ordering, and separately report the audited physical endpoint
distance. Save actions, RGB frames, proprio tokens, model identity and
resource counts. Position evaluation is post-action and does not affect
selection. Frozen model inference must use host CUDA.

These 36 trajectories share four already-exposed roots and do not provide
independent statistical samples. The RGB encoder also requires an estimated
proprio state; thus “actual RGB encoding” includes that state estimator and
is not privileged simulator-state ground truth. Initial velocity is assumed
zero, and step-5/10 velocity estimates depend on pixel differencing. A
ranking reversal does not by itself identify model, estimator or task-metric
error. This is not an independent test, an equal-budget benchmark, or P1.
