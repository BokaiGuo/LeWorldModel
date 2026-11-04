# v22 fixed-bank witness-prefix suffix diagnostic

The v22 fixed-bank panel is stopped; this is a separately specified
post-outcome privileged diagnostic, not a planner arm or gate evaluation.
Require the sealed panel, its trace/probe audits and the separately
recorded feasibility witnesses. For each root, choose the first witness
that already satisfies the unchanged success threshold. Freeze its first
five actions. Report the original witness terminal distance and the
zero-suffix endpoint.

Run exactly three 120-sample full-action suffix CEM starts with the same
optimizer, initial zero suffix, random seeds and per-start budget used by
the fixed-bank diagnostic for that root. The seed is shared by candidate
and witness prefix; random draws may produce different updated trajectories
because outcomes differ. Save all scored action batches and report each
best score and success. Charge the zero-tail evaluation and suffix
evaluations separately. Retain all failures; run once, no extra restarts
or hand-tuning. Witnesses are privileged evaluator-only input and never
become a formal candidate.

Replay saved batches using the alternate implementation of the same Wall
geometry and verify prefix preservation, bounds, counts and scores.
Success means this particular optimizer found a continuation; failure
cannot prove that the prefix is impossible or that more search cannot
succeed. This does not prove PC2 or independently confirm PC3.
