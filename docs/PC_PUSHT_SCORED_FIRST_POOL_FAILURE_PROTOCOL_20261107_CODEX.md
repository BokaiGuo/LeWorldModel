# PushT scored-incumbent first-plan complete-pool diagnosis

Use the already-exposed four-task seed-85 panel in
`pc_scored_incumbent_real_policy_parity_20261107_codex`. Rerun its action-only
solver with a passive recorder; save every candidate, score and predicted
successor from each model query in the first `solve` call. Check all four
first-plan selections and physical dispatch traces against the existing run.
No evaluator truth enters planning or selection.

After the pool is saved, replay every byte-distinct chain per task from the
same simulator reset state and goal, using the original action normalizer and
all 25 actions. Record first success, final success and per-packet state for
every unique action, and retain all occurrence costs and query indices.
Verify that reset physical snapshots match the real closed-loop trace. Report
pool availability of success, ranking of successful candidates, selection,
and closure outcomes separately; distinguish open-loop ever-success from
closed-loop success. Do not treat duplicate chains or tasks as independent.

This is an exposed development diagnosis, not a performance comparison or gate
confirmation. Simulator reset state, true success and counterfactual outcomes
are evaluation-only. Do not change model scoring, stop rules, selection, or
physics to obtain a desired result.
