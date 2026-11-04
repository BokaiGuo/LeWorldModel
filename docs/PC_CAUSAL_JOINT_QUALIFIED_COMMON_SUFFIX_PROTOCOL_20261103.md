# Common-suffix physical intervention on joint-search qualified prefix

The development root is 8349, selected because the preceding joint search
had its only model-qualified prefix there. The fixed prefixes are exactly
the saved `raw_qualified`, `zero`, and `greedy` five-action prefixes.
They share the saved two-action zero observation prelude.

Use the same predeclared ten-action suffix family for all three prefixes:
zero; the qualified prefix's model-generated two-packet suffix at its saved
iteration; the greedy branch's model-generated two-packet suffix at its saved
iteration; and two seeded uniform raw-action suffixes from seed 2026110349,
sampled independently in [-0.5,0.5]. The last four suffixes are not selected
using physical outcomes. Do not optimize or replan for this experiment.

Execute all 15 combinations in the RGB backend. Verify each branch's first
eight RGB frames and first seven actions against its respective previously
saved branch. Save all actions and 18-frame histories. Independently replay
the physical position backend at cp2, cp7 and cp17. Report prefix physical
distance, absolute cp17 distance and change from the same prefix's zero
suffix; also report position-vector change against the same-prefix zero
suffix. This tests how fixed subsequent actions respond to the prefix; it
does not assess each prefix's independently optimized control capacity.

This is an outcome-exposed development root chosen after the prior physical
result, not independent evidence. Shared suffixes include model plans but
do not use privileged simulator state for their generation. Reuse of a
qualified predecessor means the upstream Jacobian, gradient and suffix
optimization costs must be reported separately. No gate, independent
performance or superior search-efficiency claim is implied.
