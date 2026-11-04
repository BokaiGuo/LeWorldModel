# PointMaze affine-score spectral injection: development protocol

On the already exposed roots 1888–1895, compare the saved affine-score random
CEM arm with a new spectral-injection arm. Both use the same frozen model,
RGB-derived state inputs, affine-inverse score, four initial and three tail
CEM generations, 16 proposals per generation, innovations, physical
horizon and one real RGB feedback point. The random arm is the sealed
`pc_pointmaze_affine_feedback_cem_20261103_DEV02` result. The new arm replaces
four first-generation proposals with first packets along both signs of the
top right singular vector of the position-projection Jacobian at zero raw
action, at amplitudes 0.3 and 0.6; its second packet is zero. The rest of
the CEM update is unchanged and the arm replans its tail from its own RGB.

The Jacobian is the derivative of the predicted one-packet mean proprio
token after affine position projection, with raw action coordinates and
the same five-step packet normalization as the planner. Record both JVPs,
SVD, direction, singular values, and centered finite differences at raw
steps 0.01 and 0.001 as diagnostics, without using physical outcomes.
Ties use the optimizer's existing stable ordering. The action direction
is not interpreted as a complete reconfiguration program.

Both methods receive the same number of model-score samples, but the
spectral arm incurs two extra JVPs and an SVD; finite differences are
separate validation overhead. Save and report each category and elapsed
time; this is **not an equal-total-cost test**. No privileged state or
physical labels may enter scoring or candidate selection. True state is
used only for simulator reset and offline evaluation. Validate frozen
weights, scene, checkpoint, source hashes, normalizer, real RGB and both
physical replay paths. Save all candidates, scores, actions and frames.

The previous outcomes have already been seen; these are development roots,
not independent confirmation. Do not claim spectrum is generally useful
or useless, P1 is passed, or that a non-greedy mechanism exists. A fair
resource-budget comparison must let the baseline use the structure
overhead, include ordinary lookahead/strong baselines, and separately test
task-relevant capability interventions and independent roots.
