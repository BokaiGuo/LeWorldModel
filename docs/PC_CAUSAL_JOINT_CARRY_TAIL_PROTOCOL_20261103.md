# Joint-plan tail carry versus zero-initialized feedback CEM

On exposed root 8349 use the exact previously saved zero, greedy and
`raw_qualified` prefixes and each prefix's own saved joint-gradient
ten-action suffix. The source selected prefix and plans before physical
outcomes; this follow-up root and hypothesis are outcome-exposed.

For each prefix replay the same two-action prelude and five-action prefix
and verify all eight saved RGB frames. Reconstruct the actual cp7 RGB/state
context. Run the same 50-sample two-packet `packet_cem`, same physical action
bounds and seed `2026110380 + root`, with two separate initializations:
zero tail and that prefix's own original joint-plan tail. Each CEM explicitly
compares its initial plan with its final mean and chooses the lower predicted
task score. Save all proposals, scores, decisions and full physical
trajectories. Also execute each unmodified joint-plan tail. The zero-initialized
branch must reproduce the prior joint search branch's physical actions and
RGB frames; otherwise stop.

Each prefix thus has three fixed physical branches: old joint tail,
zero-initialized replan and old-tail-initialized replan. Compute cp2/cp7/cp17
physical position and compare predicted versus executed outcomes. Report
whether each replan changes the old action chain and whether model preference
agrees with physical outcome. The two replan arms each get 50 model samples;
the old-only branch incurs only its additional score evaluation. Their
different prior information and total upstream search costs prevent a
general equal-resource superiority claim.

This is an exposed single-root development intervention, not independent
confirmation or a common-suffix capability test. No privileged physical
state may enter the selector. Use real cp7 RGB for every planner; preserve
and separately report original plans even when the model prefers a new tail.
