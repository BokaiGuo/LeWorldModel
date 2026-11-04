# v22 saved-bank coverage diagnostic: post hoc protocol

The v22 reserve panel is stopped and its outcomes are already exposed.
This is a descriptive analysis of its sealed records, not a new test,
confirmation, or a reason to reopen or tune that panel. Perform no model
or dynamics evaluations.

For each of the eight paired roots, use the selected final chain of the
shared-bank multistart CEM as the reference and compare its first five
float32 actions with every v22 first-layer candidate and the retained
subset. Report minimum per-coordinate RMS action distance and exact
matches, both for all roots and separately for the one root whose CEM
selected chain achieved the already-frozen `<0.05` success threshold.
Record the selected reference score and all candidate/retained indices
used to calculate each minimum. Do not interpret proximity as physical
reachability, necessary coverage, or evidence that a suffix succeeds.

Also inspect all full-horizon v22 scored batches and count samples below
the frozen success threshold, whether selected or not. Report best saved
full-chain score and whether an exactly matching reference action chain
was scored. Recompute statistics directly from saved float32 action and
score arrays, verify manifest dependencies and trace hashes before and
after, and bind the output to input hashes. Fail closed on incomplete
records. Stop after this one audit; no additional planning samples.
