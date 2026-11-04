# v22 Reacher saved-response linearization: post hoc result

From the three sealed RGB decisions, 150 candidate records were examined,
including 120 with nonzero perturbation. This reuses saved predictions;
it does not evaluate actual physical responses. Identical nominal actions
across recorded batches differed by at most `4.77e-7` in the combined
chart/score coordinates; the first recorded evaluation was used.

For nonzero perturbations, the upper median of relative linearization
error for chart response was 0.498, 0.401 and 0.853 for decisions 0, 1
and 2 respectively. For the score coordinate it was 0.446, 0.437 and
0.641. Corresponding upper median absolute chart errors were 0.0506,
0.0499 and 0.1721; absolute score errors were 0.0491, 0.0491 and 0.5195.
The detailed rows retain decision, depth, actual action amplitude,
absolute and relative errors and cosine, including undefined cases.

These values are descriptive for one exposed task and its generated
candidate distribution. They show non-negligible local approximation
error at the actions actually sampled, but do not identify the cause of
planning success or failure, establish physical model accuracy, or
certify PC1. Group counts differ across depths and amplitudes.

Protocol: `PC_V22_REACHER_SAVED_LINEARIZATION_POSTHOC_PROTOCOL_20261104_CODEX.md`.
Detailed saved data:
`le-wm-main/results/pc_v22_timed_feedback_reacher_wiring_20261104_codex/linearization_posthoc.json`.
