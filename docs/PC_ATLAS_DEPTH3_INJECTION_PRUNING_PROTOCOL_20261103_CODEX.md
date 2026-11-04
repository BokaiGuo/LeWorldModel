# Depth-3 injected-chain pruning diagnostic protocol

This development-only diagnostic reuses the eight exposed exact-unicycle
goals and saved two-action gradient witnesses from the prior injection audit.
It is not an independent confirmation or a measurement of Atlas performance.

For saved actions `(a, b)`, require `b[1] == 0`. Use two bounded, privileged
three-action chains for each goal:

1. `early`: `(a, b, zero)`;
2. `delayed`: `(a, (-0.05, 0), b + (0.05, 0))`.

In exact unicycle dynamics these reach the same terminal position modulo
floating-point rounding. The delayed chain deliberately sacrifices intermediate
position score. Its continuation is a counterfactual if its node is pruned.
These are constructed edge cases, not evidence of population frequencies.

Run the actual, unchanged v2 search with depth 3, beam width 2, random seed
`42 + goal_index`, the previous compiler configuration and two random bridges.
Add each chain's next action only to the bank at its full-context digest, with
unique edge names; separately count visits. Reconstruct the three-step witness
directly and by split replay and verify exact full-context digest equality.

Observe the v2 function itself using Python line tracing immediately after its
beam assignment. Capture the actual expanded list and the actual selected beam
for the injected root branch. Fail closed if the search source no longer has
exactly the expected beam assignment, or if the observed beam is not the stable
cost-sorted slice of the observed expansion. Record each candidate's ordered
edge names, action sequence, cost, context digest, rank and whether it is
actually retained. Track the exact named injected chain separately from any
different path reaching an identical context. State explicitly whether the
third action was offered at the relevant matching context; do not assume
pruning prevents any other path from reaching the same context.

For each run report intermediate rank and actual retention, terminal
continuation/selection status, model terminal witness cost, competitor costs,
compiler attempted/returned counters and external diagnostic rollouts. The
action-generation work of the saved gradient planner is not included in the
compiler count. Root branches are not globally pruned. The two variants may
alter the search's candidate banks and random consumption and are not a
matched-compute performance comparison.

Run all 16 goal/variant cases, plus tests for trace capture, both intermediate
outcomes, context binding, split replay, bounded actions, and deterministic
repeatability. Preserve all old files and write new output exclusively.
