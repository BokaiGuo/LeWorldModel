# Guarded spectral predicted-tail carryover development protocol

Use the two existing Reacher development roots and all three sealed prefix
arms: selected, zero and same-bank greedy. Reconstruct each arm's originally
predicted-successor CEM suffix using its frozen model context, same seed and
budget, and require the recovered model cost to match the archived selector
receipt. The historical selector did not save suffix parameters; recovery is
a reproducibility operation, not evidence that the old runner used the tail.

Execute each sealed prefix from its original root; require exact raw-prefix
and observed RGB/action-history identity with the original physical run.
Then on real successor RGB, run the same 16-sample, 2-iteration, 4-elite CEM
as the old physical comparison, inserting this arm's predicted suffix as
candidate 1 in both iterations. Candidate 0 is the iteration mean. Other
samples follow the same seeded draw, and the final iteration's best candidate
is executed. This substitutes one random sample, not an extra evaluation.
All arms have the same feedback and suffix budget; structure generation,
all selector calls and source initial search remain part of their actual
costs, and full equal-compute certification is not claimed.

Inputs and source reports are SHA-bound; checkpoint weights are checked before
and after. Report model and real physical outcomes, candidate choice, retained
actions, source hashes and cost limitations. Roots and existing outcomes were
already observed, so this is exploratory and cannot pass P1 or PC gates.
