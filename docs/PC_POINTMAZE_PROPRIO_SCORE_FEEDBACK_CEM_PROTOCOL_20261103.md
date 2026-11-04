# RGB-derived goal proprio score in feedback CEM

Fix all eight exposed development roots 1888–1895 in the previous
`PC_POINTMAZE_CAUSAL_EXTRA_CEM_BUDGET_PROTOCOL_20261103.md`; do not choose
roots using outcomes. Evaluate two random-CEM arms, visual score and
predicted-proprio score, using identical saved per-root innovations: four
batches of 16 initial two-packet candidates and three batches of 16 tail
one-packet candidates. The initial state's position comes from RGB with zero
velocity. At cp5 use the arm's own real cp4/cp5 RGB-derived position and
causal velocity, then replan its own tail. Both arms execute five steps per
packet, without early stopping. The target is frame-10 RGB; predicted-proprio
goal encoding uses position estimated from target RGB and zero velocity.

The frozen model, scene, camera, original random CEM, action scaling and
normalizer are unchanged. The physical state may be used only for reset and
post-execution evaluation, never for planning or selection. Enforce hashes
for the source artifacts and checkpoint and reproduce the prior visual arm's
full selected plans, feedback/terminal RGB, and endpoint on every root.
Run the prior ten-step legacy-render and position-backend replay checks for
both arms. Save all plans, scores, complete CEM traces, inputs, RGB frames,
action histories, timing, model parameter hashes, and prediction sample
counts. Keep errors visible, never replace a root or select a favorable arm.

Report each root's strict offline terminal position distance and `<0.5`
indicator. This is a fixed-budget objective ablation, not a verified
equal-total-compute comparison: count encoding and prediction calls and
timings separately; extra scoring arithmetic and timing may differ. Roots
were already exposed and training membership is unknown. Do not claim native
online success, independent generalization, spectral advantage, causal
reconfiguration, or P1 admission from this experiment.
