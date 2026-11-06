# v25 fixed-bank suffix diagnostic: development protocol

Before execution, use exactly four exact-Wall tasks with start
`(-0.5, y, theta)`, `y ∈ {-0.21, 0.21}`, `theta ∈ {-1.19, 1.19}`
and goal `(0.5, 0.16)`. Reject exact matches with the enumerated panels
used by the v25 retention runner and that runner's own tasks. This is a
limited overlap check; broader provenance is unknown.

Run v25 response with 1,920 prediction tokens and freeze its first-layer
candidate bank and same-bank greedy, state-diverse, response-value and
random retention sets. Compute ordinary lookahead by scoring 24 seeded
uniform full-action tails per prefix, together with each prefix's zero
tail; retain the two immediate-greedy slots and fill remaining slots by
lookahead rank. Write a hash-bound selection record before any common
suffix optimization. Selection does not inspect later suffix outcomes.

On every fixed prefix, run the same full-action CEM with two restarts,
121 samples per restart, batch size 12 and four elites. Restart seeds
are shared across candidates. Report the best observed score per prefix,
the best per retained set, its gap to the all-bank best, and the original
success threshold `<0.05`. Explicitly report which response-added and
response-displaced candidates are compared. This finite search is not a
certificate of the true optimal suffix; report it as a diagnostic.

Count generation, lookahead, and all-candidate suffix prediction tokens
separately. Generation includes v25's own successor probes and full-chain
refinement. All-candidate suffix evaluation intentionally exceeds the
full-method budget; it is not a PC3 resource comparison. Persist each
prediction attempt, action, returned result or failure before proceeding.
Audit hashes, fixed-prefix mapping, sample counts, selection records,
scores and alternate same-environment physical replay. Record failed roots
without replacement, stop after the four roots, and do not tune this panel.
This diagnostic does not establish the PC2 same-chain mechanism.
