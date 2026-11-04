# Predicted-proprio spectral CEM development comparison

Use all exposed roots 1888–1895, the frozen checkpoint, RGB state/goal
projection, cp4-to-cp5 causal RGB velocity, and the saved initial/tail
innovations from the previous causal-CEM run. Both arms use the same
predicted-proprio goal-token score, the same three batches of 16 initial
two-packet CEM candidates and three batches of 16 one-packet tail candidates.
Execute each arm's first packet, observe its own real RGB, then replan and
execute its own tail. All branches run ten steps; no early termination.

For the spectral arm only, compute the two-column Jacobian of the predicted
one-packet proprio token with respect to the two raw action coordinates at
zero, using two JVPs. SVD it and replace the first four first-generation
two-packet CEM proposals with first-packet actions `+.3, -.3, +.6, -.6`
times its top right singular vector and zero second packet. The random arm
uses the same unmodified innovations. Do not select direction or roots using
outcomes. No simulator state is used in planning; it is allowed for reset
and afterward for evaluation. No training or privileged future input.

Count samples, JVPs, SVDs, RGB encoding and elapsed times separately;
identical sample counts are not equal total computation. Replay all actions
through the legacy RGB renderer and position evaluator and check matching
terminal RGB and positions. Save traces, Jacobians, frames, actions and
endpoint metrics; fail closed on non-finite inputs or artifact mismatches.
Report all roots, including strict offline distance `<0.5`, not online
success. These are previously examined development roots with unknown
checkpoint training provenance, not independent evidence. This is a
spectral-initialization ablation, not proof of reconfiguration, composition,
P1 admission, strong-baseline advantage, or total-budget fairness.
