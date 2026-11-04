# Q-consistent prefixes: common-suffix development intervention

Before execution, fix the two roots and all six completed arms in
`reacher_pc_q_consistent_physical_20261102_v2/report.json`, SHA-256
`0671a015c0b850ab41fc170817199c2658b5a57e5c5418de4042ffc32dd2474b`.
Use every selected, zero and same-bank Q-greedy five-step prefix and every
saved optimized ten-step suffix from those three arms, plus a ten-step raw
zero suffix. Apply the complete 3 × 4 cross-product per root. Do not select
suffixes based on outcomes; the suffixes and these development roots were
already outcome-exposed, so the result is exploratory.

Restore the same full physical root for every branch; verify the root and
goal RGB and history index against the sealed model source. Bind the source
and root reports by SHA-256 and check prefix action hashes. Replay the
recorded after-prefix distance with absolute tolerance 1e-8. Save every
physical state, reward and stop; preserve failed and censored branches rather
than imputing values. Compare absolute terminal qpos distance and each
suffix's change against the same prefix's zero suffix. If exact duplicate
tails occur, report them as duplicates, not independent interventions.

This physical intervention runs no neural inference and never uses physical
state to choose actions. It does not establish the native success metric,
equal total compute, independently optimized control advantage, independent
generalization, or P1 admission. Outcomes cannot be used to rewrite this
protocol or silently change the prior physical run.
