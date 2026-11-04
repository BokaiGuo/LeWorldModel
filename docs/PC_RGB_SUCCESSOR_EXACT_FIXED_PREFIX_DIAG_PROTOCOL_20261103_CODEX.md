# Fixed-prefix suffix-search development diagnostic

This is a post-outcome diagnostic on all 24 already exposed wall tasks, not a
new planner or independent confirmation. It does not modify the stopped
six-dimensional block-program experiment. Its question is whether the
existing actual successor-prefix bank or two privileged known-feasible
prefixes support finding good suffixes under the same stronger suffix search.

For every task take all four recorded 10-step successor prefixes and the
first 5 and 10 steps of the existing shortest analytic feasible witness.
The latter use privileged geometric information and are isolated diagnostics,
never planner inputs or method performance. For each of these six prefixes
run the same seeded `optimize` CEM with 384 complete-chain evaluations,
maximal batch 12, same root-specific seed `2026121500 + root`, and record
the 192- and 384-sample incumbent. Record every scored action and value.
Recompute all traces using the original exact wall backend and verify prefix
binding, bounds, scores, allocated counts, and both checkpoints.

Report per-candidate and per-root success `<0.05`, and replay the unmodified
full analytic suffix separately to verify it really completes each witness.
The 384-evaluation resource is for a quality diagnosis, not equal to the
original full planner budget and not evidence of a fair performance win.
If a CEM search fails with a known feasible suffix, distinguish search
coverage from prefix infeasibility. Do not retune, add seeds, select roots,
or change the stored successor results based on outcomes.
