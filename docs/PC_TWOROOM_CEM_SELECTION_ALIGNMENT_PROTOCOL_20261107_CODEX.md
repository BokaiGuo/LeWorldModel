# Equal-CEM selection alignment: post-outcome development audit

The prior CEM and physical outcomes are already exposed. Audit every one of
the 40 roots, for the original and reversed prefixes. For each arm compare
the CEM-selected tail with the same previously selected B tail, which was
actually present in the first CEM population. This is 80 paired comparisons,
not an independent test; pairs sharing a root are not independent tasks.

Reconstruct actions from sealed artifacts, verify B-tail membership, action
equality and frozen source/input fingerprints, and replay both chains from
the same reset. Verify all physical step distances and success/truncation
against the two earlier physical artifacts. For each chain record the
five-packet model rollout and same-time frozen-encoder representation of
the real rendered state, endpoint goal costs, and full physical trajectory.
Check re-evaluated endpoint cost against the CEM's sealed cost, allowing
1e-3 absolute and 1e-5 relative tolerance for batch-dependent floating point.

Summarize all pairs, including pairs where both chains succeed. Separately
report the already-known cases where the CEM chain fails and B succeeds:
model predicted ordering and real-encoding ordering. The success predicate is
ever-success; endpoint embedding cost is an endpoint metric, and neither
agreement nor disagreement alone establishes a root cause. No true post-action
image, state, or label enters the planner; this audit makes no new selection,
gate decision or generalization claim.
