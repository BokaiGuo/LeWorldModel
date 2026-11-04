# v22 successor-response retention: fixed-bank E0 development protocol

Hypothesis: on a common automatically generated v22 candidate bank,
retention based on successor response/ridge value may retain better
optimizable delayed-reward prefixes than immediate greedy or state-only
diversity. This is an exploratory mechanism diagnostic, not PC2 or PC3
confirmation. A negative result is valid.

Use exactly four exact-Wall roots with start `(-0.5, y, theta)`,
`y ∈ {-0.25, 0.25}`, `theta ∈ {-1.14, 1.14}`, goal `(0.5, 0.22)`.
Run the v22 compiler once per root, budget 1,920 prediction tokens, and
freeze its ten first-layer candidates before evaluating retention rules.
Reject overlap with the explicitly cited prior panels. Broader root
history and training provenance remain unknown.

For each fixed prefix, evaluate 41 centered finite-difference successor
probes at step 0.05; compute the predefined ridge value. Evaluate 24
seeded random-lookahead tails. Independently optimize each prefix using
the same full-action suffix CEM with three starts and 120 samples per
start. Apply the existing fixed-bank greedy, state diversity,
response/ridge, lookahead and random selectors without using suffix
outcomes. Report all-bank and selected-bank best distance, success using
the frozen `<0.05` threshold, and candidate indices; no post hoc success
threshold or strategy change. Keep failed roots in the denominator.

Report compiler, probe, lookahead and suffix costs separately. The
all-candidate diagnostic spends substantially more than 1,920 tokens and
is not an equal-total-compute method comparison. Replay saved trajectories
with the alternate implementation of the same environment, verify
candidate mapping, selectors, suffix outcomes, action bounds and ledger.
Stop after four roots; do not replace failed roots or rerun/tune this panel.
