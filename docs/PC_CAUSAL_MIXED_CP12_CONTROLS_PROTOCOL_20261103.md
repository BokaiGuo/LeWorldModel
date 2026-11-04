# Matching cp12 feedback for zero and greedy controls

Apply the identical cp12 procedure from
`PC_CAUSAL_MIXED_CP12_REPLAN_PROTOCOL_20261103.md` to the two fixed zero and
greedy trajectories in the parent mixed-prefix result, across the same four
exposed roots. Replay and verify saved RGB through cp12, initialize a
16-by-3-iteration CEM with the original five-action tail, use the same
per-root random seed and frozen task score, then execute its selection.

Record complete physical trajectories, CEM populations and scores, cp17 RGB
score and independently replayed physical position. Compare these results to
the 32 candidate branches which already received identical cp12 feedback.
All comparisons are exploratory: the candidates and roots have been exposed;
the upstream raw/spectral/zero/greedy generation and search costs are not
equal. Neither physical outcome nor simulator state enters planning.
