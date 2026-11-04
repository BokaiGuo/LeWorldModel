# Wall triprefix common-tail and equal-CEM development diagnostic

This is a post-outcome follow-up on one previously exposed exact task, not
an independent or physical learned-model experiment. Fix the three prefixes
to the bound-normalized prior receipt for seed 20261103: selected spectral
`extra_minus`, zero, and shared `progress`. Confirm all three are four bounded
actions and bind their source receipt by SHA-256. Their prefix-time error and
position must be reported; do not describe their initial states as equal.

For each of seeds 20261103, 20261104, 20261105, independently optimize a
12-action suffix for each prefix using the same exact wall dynamics and
position-distance objective. Use 4 rounds × 64 candidates, 8 elites,
zero initial mean and coordinate standard deviation `HIGH/2`, clipping before
both scoring and elite fitting, minimum coordinate std `1e-5`. Use a fresh RNG
with the same seed for each arm, providing paired standard-normal draws;
subsequent transformed candidate arrays may differ as the distributions
adapt. Fix the prefix and sample every complete suffix, keeping the earliest
lowest-cost candidate across all rounds. Count suffix dynamics transitions
and compute elapsed time separately. Record all best suffixes, histories,
selected prefix, prefix/final position and distance, blocked moves, and
direct replay. Include the four prefix steps and verification separately in
the cost report; this does not account for original spectral generation and
is not a full-method equal-total-compute claim.

Additionally for each seed apply the exact suffix selected by the spectral
arm to each of the three prefixes, as well as a fixed all-zero suffix. Report
absolute terminal distance and change relative to each prefix's own
zero-suffix terminal distance. This common-suffix intervention tests effects
of the prefix on that selected action, not independently optimized control
capacity; the spectral suffix is optimized for the spectral state. Separately
report each arm's own optimized result. The model here is the exact backend;
no privileged simulator state is fed into an RGB planner.

The three optimization seeds are repetitions of one task, not independent
roots. No arm is chosen from outcomes, no gate changes, and no generalization
or real-physics claim follows. Record full source and protocol hashes, use
exclusive output creation, preserve all previous results, and run replay,
action-binding and count tests.
