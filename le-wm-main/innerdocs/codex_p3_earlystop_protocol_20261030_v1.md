# P3 early-stop exploration, fixed before this run

The known two-root roster, candidate bank, three arms, allocation rule, axes
and per-point caps are inherited from the resolved exploration. Each allocation
is a maximum, not a target. For each root evaluate in bank order; after the
first incumbent, stop immediately on five consecutive candidates whose model
cost does not strictly improve the incumbent. Ties count as non-improvement;
an improvement resets the counter. If the fifth non-improvement coincides
with the last allowed candidate, patience takes precedence in the stop label.
Every completed candidate is charged, and unused capacity is never padded.

The six slots and 30-second launch-to-exit deadline are fixed before running;
consumer execution is outside that deadline. A failed, timed-out or missing
slot stays in the denominator with unknown unobserved consumption; no retry.
All call axes are separate, including nested calls; they are not summed into
FLOPs. The cooperative GPU lock is not a global isolation guarantee.
This rule is a post-observation engineering exploration on known roots, not
independent confirmation, a claim of equal total compute, or a scientific pass.
