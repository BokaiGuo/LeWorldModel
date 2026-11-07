# Distinct-prefix retention: development protocol

Use the 40 exposed RGB roots and regenerate the original 13 prefixes and nine
tails. Before any model query, collapse prefixes whose complete float32 action
arrays are elementwise equal, preserving the first occurrence and recording
the original-index mapping. Fail closed if fewer than two distinct prefixes.
This is an engineering correction to candidate identity, not an assumption
that similar but unequal actions are interchangeable.

All arms use the same distinct pool and scores: ordinary A-probe lookahead,
the original anchor-plus-effect-vector diversity rule, and an identity-shuffled
vector rule. Draw the shuffle deterministically per root over the distinct
pool. Query five A tails for every distinct prefix, then separately query four
B tails for each arm's two retained prefixes. No truth or unretained B score
enters selection. Count actual model chains and synchronized score time; do not
claim full-task wall-clock or a cost advantage from nominal query counts.

Seal decisions and original action indices before opening the existing
physical replay. Reuse that replay only after validating fingerprints, root
and exact action arrays; preserve all roots and report selected-chain success,
retained B-useful prefix counts, duplicate-free retained slots, and query
counts. Any difference from the old method is developmental and confounded
by deduplication, batch layout and shuffle; it cannot establish an independent
method advantage, unqueried-tail generalization or a gate pass.
