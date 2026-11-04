# CP12 history task-score rank diagnostic: frozen protocol

Freeze this protocol before computing the new task scores or rankings. Input
is the saved 32-branch single/three-frame prediction receipt and tensors
`pc_pointmaze_cp12_three_frame_same_action_20261103_CODEX`, on roots 8343,
8345, 8349 and 8351, eight ordered branches per root. No new action, model
prediction, planning, simulation, selection or sample filtering is allowed.
The source clock is cp2/cp7/cp12 -> cp17; the arms share the cp12 outgoing
five commands and actual cp17 target.

Use the existing cp12 refresh projected task score exactly: extract the RGB
goal from the same obstacle bank, infer goal position from the same camera
and scene calibration with zero goal velocity, normalize with the same
proprio statistics, and apply the checkpoint's affine proprio encoder.
Use the same pseudoinverse and float32 Torch scoring kernel as cp12 refresh.
The goal token is reconstructed from the checkpoint's affine layer, rather
than rerunning the visual network; its task-score validity must be checked
against all 32 old one-frame and actual-RGB scores with `rtol=3e-5`,
`atol=1e-6`. Bind the old source, goal bank, calibration, checkpoint,
prediction receipt and all NPZ inputs by hash; reject mismatched roots,
branches, model identity, nonfinite arrays and invalid dimensions.

Write a score-only receipt and per-branch arrays first, without consuming
physical outcomes or choosing actions. Then an evaluation stage verifies
that receipt and reads the old refresh receipt's actual cp17 RGB-derived
scores and evaluation-only physical distances. For each root and each
prediction arm, count all 28 within-root pairs, strict reversals, ties in
each variable and comparable pairs, against both actual RGB-derived task
score and physical distance. Recompute actual RGB task scores from saved
actual cp17 proprio latents and check against the old receipt. Require the
one-frame reversal counts and tie counts to equal the old refresh receipt;
the old receipt's scores are never fitted or used to adjust scores.

Scores and rankings describe saved outcome-exposed development trajectories;
they are not independent samples, a fresh experiment, closed-loop success,
or evidence that one planner is better. RGB-derived velocities are interval
averages and historical timing/provenance remain uncertain. Prior inference
cost was 32 one-frame plus 32 three-frame predictor samples, 128 observation
encodes and no new physical steps; three-frame history requires more
context compute. This stage uses 96 projected score evaluations, one
analytically reconstructed goal token, no predictor samples or image encodes,
no new physical steps and no candidate selection. Physical distances are read
only after scores are sealed and are never supplied to a planning procedure.
