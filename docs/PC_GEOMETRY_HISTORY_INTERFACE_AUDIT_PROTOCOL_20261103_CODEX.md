# Same-action history-interface comparison, exposed development panel

The paired 48 saved model-optimized chains and their native physical
endpoints are fixed before this diagnostic. No optimization, candidate
modification, score tuning, or model parameter update is allowed. The
checkpoint config declares `frameskip=5`, `action_skip=1`, `num_hist=3`,
`ctxt_window_train_rollout=3`; the data loader concatenates five normalized
raw actions per transition. Training code uses the observation/action
sequence and next-frame loss. This supports comparing three real frame
contexts with two preceding true action packets. It does not by itself
prove one-frame inference is invalid or reveal checkpoint provenance.

For every existing setup root execute 15 actual zero actions in the
official native RGB bridge and preserve all resulting RGB frames. Construct
causal visual-velocity observations at steps 5, 10 and 15, each from the
immediately previous and current frames. Bind the true five-zero-action
packets from 5→10 and 10→15, normalized coordinatewise using the existing
dataset action mean and standard deviation, in chronological order. Check
that current RGB matches the saved original one-step-warmup RGB exactly,
and that native position and velocity remain unchanged through 15 zeros;
otherwise mark the comparison unmatched and stop.

For each fixed 15-action candidate chain, compare (A) the previous
single-current-frame, empty-packet-history, window-one implementation and
(B) these same current RGB/goal and candidate actions plus the two older
RGB contexts and their two true history packets, window three. Predict
three five-action transitions in both arms. Verify A's terminal projected
goal-distance scores reproduce the original receipt within `1e-5`. Record
both projected cost predictions, their absolute error against the saved
physical position distance (different measurement spaces), all within-root
pairwise ordering reversals relative to physical distance, selected index
and changes. Score ties explicitly and retain all roots. This is an exposed
development comparison, not an independent test or proof that B is the
unique training-correct deployment context. A single three-frame history
comparison may not distinguish historical action input from visual context
or attention window effects.
