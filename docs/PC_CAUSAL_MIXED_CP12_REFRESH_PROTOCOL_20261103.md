# Fixed-action cp12 refresh diagnostic

On all 32 saved, outcome-exposed trajectories in
`pc_causal_mixed_rejected_physical_20261103_DEV01`, reproduce the true cp7,
cp12 and cp17 RGB-derived state from the two latest real frames at each
time. Use the same frozen model, action normalizer, affine task score, goal
and pixel preprocessing as the original runner.

For each trajectory predict cp17 twice using the same saved physical actions:

1. open-loop from cp7 with both five-action packets;
2. refresh at cp12 from its actual RGB/state and predict with the second
   packet only.

Check the open-loop score against the saved feedback score (the model had
scored the same physical tail), and check action bounds. Encode actual cp17
RGB/state independently. Save predicted and actual visual/proprio embeddings,
task scores, per-field mean squared errors and task-score absolute errors.
Compare predicted task-score ranking with actual RGB-derived task-score and
physical position distance inside each root, including ties. Compute MSE
paired differences and ranking changes separately; lower MSE is not assumed
to improve ranking. Physical state is evaluation-only.

No new physical actions, candidate selection, optimization or root outcomes
are permitted. Existing candidate and root exposures prevent independent
confirmation. This differs from a deployable replanner because the future
actions are fixed by past experiments; cp12 refresh does not alter a plan.
Action history is saved, but this checkpoint adapter encodes the latest
frame and RGB-estimated state, not the complete executed action history.
