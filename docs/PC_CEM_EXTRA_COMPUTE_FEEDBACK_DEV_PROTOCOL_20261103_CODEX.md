# Extra-resource direct CEM: paired development protocol

This is a post-outcome development follow-up, not a preregistered
confirmation. Reuse the same four exposed PointMaze roots, goal, checkpoint,
warmup, RGB/action-history interface, three feedback times, 15-step model
horizon, five-action execution packets, CEM optimizer and seeds as the
matched-token panel. The original direct CEM used 132 prediction tokens per
decision (44 full-horizon samples). Run two additional direct-CEM arms:
144 tokens (48 samples) and 156 tokens (52 samples) at each decision.

The two extra budgets are fixed before execution and both must be reported,
not chosen after seeing outcomes. They give CEM additional search opportunity
to probe the effect of goal-mode structural overhead; no conversion from
SVD, differentiation, encoding or wall time to token samples is asserted.
Separately report measured prediction calls/samples, encoding, planning
wall time, SVD and shared setup. Do not claim total-compute equivalence or
independent significance. Seal each decision before dispatch, then physical
replay each entire executed trajectory on the same backend, scene and goal.
Retain all roots and arms.
