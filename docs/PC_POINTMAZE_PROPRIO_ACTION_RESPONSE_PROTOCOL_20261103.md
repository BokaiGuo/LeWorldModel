# Privileged initial-proprio action-response diagnostic

On the same four exposed roots and eight 48-candidate banks as the sealed
action-response report
`a0_pointmaze_cem_action_response_20261103_DEV01.json` (SHA-256
`0f7492d9be9c39d07ee4805c359c252f50a354d20214ce1f4634fce39023b471`),
compare the original raw-zero proprio context with the archive's float32
initial four-dimensional state supplied as raw proprio. This is a
privileged diagnostic, not an RGB-only method. All candidates, RGB inputs,
goal, initial task mask, action normalization and frozen checkpoint remain
identical. Encode both contexts and fail if their visual features differ.
Use the previously saved actual RGB task-mask features and physical distances;
check all source and model hashes.

For each arm, compare centered feature response cosine and norm ratio,
task-mask prediction MSE, goal-cost pairwise reversals against actual RGB,
and the physical replay distance of the model-selected candidate. Predicted
selection is computed only from model costs; the saved physical distances
are evaluation-only. Save all predicted feature arrays and scores to allow
independent arithmetic replay. No training, model adaptation, root selection
or additional physical interaction is allowed.

The archive state is valid only as an observation at the source initial
state, and its use is prohibited in the deployed RGB planner. This diagnostic
cannot certify checkpoint training provenance, independence of the archive
normalizer, candidate-level independence, real closed-loop control, or P1.
