# Cross-prefix replay of saved waypoint suffixes

This post-outcome development diagnostic uses the entire saved 3-seed
triprefix waypoint receipt, not a newly selected subset. For each seed use
the saved winning 12-command suffix from each of the selected, zero and
progress waypoint arms. Each was generated using privileged exact pose and
known unicycle kinematics; some are unsuccessful. Apply each of these three
unchanged suffixes to all three unchanged saved prefixes, and also apply an
all-zero suffix to each prefix. This is 27 cross-prefix plus 9 zero-tail
exact-dynamics replays. No planner selects actions, no new optimization
occurs, and no physical or learned visual model is called.

First require each source replay to start with its declared prefix and match
its stored terminal distance and complete trajectory when replayed. Verify
the three suffixes and all prefixes have finite values, correct shapes and
bounded actions. Save full replay trajectories, per-step blocked flags,
suffix index of first wall crossing if any, terminal position, absolute
goal distance, zero-tail distance, and signed distance difference to that
prefix's zero-tail result. Report all 27 entries, including source-arm
replays and tied/failed suffixes; the three seeds are repeated searches on
one exposed task. Count full-chain state transitions including verification
separately. Bind original receipt and code hashes, verify inputs unchanged
across execution, and write a new exclusive output.

The suffix source arm is an outcome-exposed diagnostic, not an independently
fixed action family or a planning method. Even a common-suffix advantage does
not by itself establish better equally optimized controllability, and the
prefixes have different positions. Do not change any gate or historical
artifact; no physical, independent or RGB-only inference follows.
