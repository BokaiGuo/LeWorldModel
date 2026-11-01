# Fixed four-arm opposite-side control comparison

Frozen before control outcomes: structured, native, random, immediate_progress,
all four arms on the single V3 eligible known-root target. No outcome-dependent
arm omission or added seeds. Native evaluator and physical guards unchanged.
This adaptively constructed task is developmental, never independent evidence.

Reuse the normalized-action-gradient/full8x8-tail-exchange search exactly:
six optimizer updates, .05 peak action step, initial prefix protection,
true RGB feedback at 5/10/15, old-tail versus new-plan rescoring and existing
node accounting. Goal-only co-metric is measured for the new target and its
construction cost is reported separately. Structure costs remain charged;
equal optimization opportunity is not identical actual total cost.

Only target RGB/embedding changes, with an explicit reverse-target schema.
Witness commands, intermediate physical states and success outcomes do not enter
candidate generation, initialization, optimization or selection. Root/goal
states are evaluator inputs, not planner observations.

Retain H3/H2/H1/H1 planning horizons. This may be insufficient to imagine the
full20-step witness; do not silently extend it mid-panel. A later H4 design must
be separately declared and fully costed. Physical runs stop on native success,
truncation or domain failure; never pad a stopped trajectory to20.

Report every arm's stopping status, observed terminal metrics, prefixes and
actual computational costs. Neither scripted task eligibility nor one model
success establishes non-greedy capability change or P1 admission.
