# TwoRoom two-stage query reproduction: development protocol

This is a computational reproduction on the 40 previously exposed RGB TwoRoom
roots, not an independent test or an efficacy claim. Use the prior fixed
candidate construction and retention rules without tuning.

For each root, regenerate 13 prefixes and nine common tails from public RGB.
Score the 65 prefix/A-tail combinations in batches of at most 32. Retain two
prefixes per arm using only those scores and the fixed root permutation. For
each arm separately score its eight prefix/B-tail combinations and choose the
minimum score with stable array-order tie-breaking. No other B score or
physical label is available to the decision. The three arms share the A
queries, but the B queries are not deduplicated.

Record batches, per-stage synchronized elapsed time, costs and selected actions.
Each isolated arm needs 73 complete-chain model evaluations (365 five-step
packets); the combined experiment makes 89 evaluations (445 packets) per root.
Timings exclude model loading, data loading, environment execution and closed
loop; they are not full-system cost comparisons. Fingerprint public inputs,
model, normalizer and code. Abort on drift or invalid actions.

Only after new decisions are sealed, compare to the prior full-matrix plan:
candidate action arrays, A scores, selected B scores, retention and final
selection. Physical trace reuse requires identical root, candidate actions,
and verified prior plan/replay provenance. Changed selections are evaluated
only if the corresponding identical action is present in the prior replay.
Report numerical deviations and selection mismatches, including near ties.
This reproduction cannot establish descriptor benefit, mechanism, unseen
generalization, feedback performance, or any PC0–PC5 gate passage.
