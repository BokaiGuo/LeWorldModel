# Frozen predicted-proprio goal scoring: fixed-bank development

On all 12 sealed initial/tail banks from the 1889, 1893 and 1895
outcome-selected development roots, use the same 624 candidate programs,
current RGB inputs, causal RGB velocity at cp5, frozen checkpoint, action
normalization and model rollout as the source planner. For the goal, use
the existing RGB marker projection and set unobserved goal velocity to
zero. Encode this RGB-derived state with the frozen proprio encoder.

Compare the mean squared distance of the predicted terminal proprio token
to the encoded goal proprio token against the original predicted visual
score. Compute the model-derived scores and save them before looking up
sealed physical outcome distances. Then, separately for each bank, report
each selection's physical distance, physical oracle distance, paired rank
reversals and ties. Recompute the original visual scores and fail closed if
they disagree with the sealed bank within rtol 1e-4, atol 1e-6. Save
scores, goal encoding, input hashes, model hashes and prediction sample
counts. No model training, ground-truth state in scoring, candidate
selection by outcome, physical reruns or gate modifications are allowed.

This is an adaptive, exposed-root fixed-bank evaluation, not a prospective
control test. The physical oracle and physical outcome are evaluators only.
The projected goal comes from observed RGB and static geometry, not future
RGB; goal velocity zero is an explicit assumption. A positive result does
not establish a causal mechanism, equal-compute advantage, closed-loop
performance, or independent confirmation.
