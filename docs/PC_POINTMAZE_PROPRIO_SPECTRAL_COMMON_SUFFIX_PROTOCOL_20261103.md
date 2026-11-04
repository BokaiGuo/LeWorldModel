# Common-suffix intervention for saved spectral prefixes

Fix all eight exposed roots 1888–1895 from the sealed
`a1_pointmaze_proprio_spectral_cem_20261103_DEV01` report. From each same
reset, test three five-step prefixes: saved random, saved spectral and zero.
For each prefix test the same three five-step suffixes: zero, the saved
random arm's optimized suffix, and the saved spectral arm's optimized suffix.
This yields 72 ten-step deterministic physical trajectories, including
duplicates, without model inference or outcome-dependent candidate choice.

For each trajectory save actions, legacy terminal RGB and full endpoint,
and position-backend cp5 and cp10. Check exact position-backend versus
legacy cp10 positions and reproduce the two original same-arm terminal RGB
and full endpoint arrays. Record initial, cp5 and cp10 physical position and
distance to the archive's frame-10 goal as evaluation-only. Compute each
suffix's endpoint displacement relative to the same-prefix zero suffix,
and record all distances and strict `<0.5` endpoint indicators. Preserve
root/prefix/suffix identities and do not select favorable roots.

This exposed-root physical intervention addresses the effect of the same
suffix after different prefixes, not whether each state is equally
replannable or whether non-greedy reconfiguration occurred. In particular
the prefixes here are previously chosen optimizer outputs, not guaranteed
non-progress actions. The initial state is used only to reset physics;
simulator state and all counterfactual outcomes are forbidden from entering
a planner. No new planning, checkpoint inference, independent root,
equal-total-budget advantage or P1 admission is claimed.
