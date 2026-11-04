# Fixed-pair goal-Q attribution

On the 26 previously exposed L2-versus-direction suffix pairs, apply the
existing goal-image encoder co-metric to both frozen-model terminal
predictions and replayed final RGB encodings. Use the same saved cp5 RGB,
goal, normalized suffix packets and actual cp5/cp10/cp15 RGB and physical
action history as the existing fixed-pair attribution. Use the previously
authenticated goal-only Gram for each root, with its existing ridge and
scoring code; no re-estimation, metric selection, candidate selection or
new physical execution is permitted.

Persist target, predicted and observed coordinate vectors, Q scores, and
source identifiers to permit independent score recomputation. Report rank
reversals and exact ties separately by root and keep physical outcomes
evaluation-only. This is a post-exposure development diagnostic on dependent
pairs, not a deployable selector evaluation, independent test, mechanism
proof, or P1 admission. Metric construction cost is not a planner budget.
