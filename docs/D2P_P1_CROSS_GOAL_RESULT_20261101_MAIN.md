# Cross-goal development panel: completed attempts

All eight predefined slots now have actual results. Seven completed20; root0
native left the canvas at step2 (pusher y=516.27735) and was correctly stopped.
Native success is 0/8 attempted. The failed slot is retained, not replaced.
The original batch stopped after that failure; six previously unattempted slots
were executed independently with the exact same method and task pairing.

Pairing: root0→saved target1; root1→saved target0. Method is the fixed-prefix,
response-diverse, first-feedback-carry variant. No model training or CPU neural
fallback. Both roots/targets are known development assets. Twenty-step
reachability of the cross pairs is unknown; this panel cannot establish
theoretical impossibility or independent generalization.

## Concrete observed disconnect

Root0 structured never changes object pose through20, while pusher goal error
falls from328.458 to175.498. Its actual RGB latent goal MSE falls from1.73620
at0 to1.47024 at15. Pusher progress is part of the native task; this alone
does not prove an incorrectly specified metric.

Root1 structured gives a stricter example at15 relative to0:
- Pusher error 169.07859→174.25819 (worse).
- Object position error 134.42333→145.85931 (worse).
- Wrapped object angle error 1.03495→1.31158 (worse).
- ACTUAL RGB latent goal MSE 1.41679→1.11000 (better).

These observations use real encoded images, not imagined states. Thus this
particular score/task mismatch cannot be removed solely by repairing rollout
prediction. It does not imply every model prediction or interface is correct.

## Evidence and verification

Joined source hashes and per-checkpoint metrics:
`le-wm-main/results/p1_cross_goal_analysis_20261101_main_MAIN01.json`.
Analysis binds physical PNGs, latent files, observation step/image identity and
target state to the other-root target. Missing terminal embeddings stay null.
The six resumed shards passed saved action/plan/source/ledger verification:
`le-wm-main/results/p1_cross_goal_shards_saved_20261101_main_MAIN01.json`,
24 actual nodes. This validator was not used to promote the original partial
two-arm batch into a fully accepted report.

Next exploration fixes push-and-return goals rather than silently removing
pusher error from the task. Goal synthesis must first reproduce original
authenticated images exactly; reset/state restoration failures are engineering
errors, not new P1 evidence. All historical outcomes and original gate remain.
