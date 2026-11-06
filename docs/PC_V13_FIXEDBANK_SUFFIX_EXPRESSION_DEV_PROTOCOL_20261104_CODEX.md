# Fixed-bank suffix expression comparison: development protocol

Hypothesis: four-knot interpolation may limit continuation quality even
when the candidate prefix is fixed. On a new four-task exact-Wall
development panel compare four-knot against per-step (ten-knot) CEM.
Both use the same candidate prefixes, zero-suffix incumbent, three
480-proposal starts, batch 12, elite count 4, scale 0.35, scale floor 0.07,
and bounded raw actions. The seed for a root/start is
`2029152000 + 10*root + restart`, shared between the two arms, although
the different parameter dimensions prevent matching individual proposals.
No response-only arm receives extra suffix samples.

Tasks are the Cartesian product of starts `[-0.5,y,theta]`,
`y ∈ {-0.19,0.18}`, `theta ∈ {-0.7,0.8}`, and goal `[0.5,-0.17]`.
Explicitly compare against the prior prospective and budget diagnostic
task sets; broader source independence remains unknown. Generate the bank
using unchanged v13 with `seed=2029151000+root`, 1920 tokens, and retain
all ten actual first-layer candidates. Both arms score the same fixed bank;
do not inject reference paths.

Record all compiler and suffix prediction calls, exact float32 action
batches, positions, scores, distribution updates and resource counts.
Report each candidate and each task, all-bank and actual-retained subsets,
with existing 0.05 success threshold. The comparison is a diagnostic
with substantial shared generation and per-candidate optimization cost,
not equal-total-budget full-method planning or PC2/PC3 confirmation.

Seal sources and protocol before execution. Stop after four tasks and 40
paired candidate comparisons; no retry, replacement, or tuning. A separate
audit checks hashes, all saved batches using alternate same-environment
geometry, packet costs, fixed prefixes, each optimization incumbent and
sample counts. Record any failure; do not infer feasibility or theoretical
invalidity from a negative result.
