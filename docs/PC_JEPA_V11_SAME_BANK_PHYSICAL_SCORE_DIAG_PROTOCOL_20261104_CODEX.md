# v11 JEPA fixed-bank physical score diagnostic

Post-hoc analysis only: use the first 64 full 15-step action chains, including
duplicates, in chronological order from the already sealed response-arm
decision-0 prediction trace of the constructed-feasible v11 task. This is a
deterministic index rule, not a new planner search, root selection, or a
representative sample. No candidate or scoring rule is changed based on
physical outcomes. The planner is never given these outcomes.

For each saved float32 chain reset the same native backend to the frozen seed,
replay the ten fixed warmup commands, verify the RGB, proprio and state exactly
against the sealed boundary, then execute exactly 15 commands. Save terminal
RGB, full states, actions, model scores, errors and failure records. Stop on
boundary mismatch; preserve all failures and the total denominator. Compare
pairwise ordering of the model-predicted terminal latent MSE against physical
joint XY error, object XY error, and wrapped object-angle error. Exclude ties
explicitly and report valid, agreeing and tied-pair counts; candidates are
correlated and this is not an estimate of generalization or significance.

A separate model-only process should encode saved actual terminal frames and
the goal with the same frozen encoder and preprocessing; compare predicted
latent MSE with actual-terminal latent MSE. Its input must exclude physical
states, success labels and witness actions. The physical score must not become
a model input or be used for plan selection. Source hashes, raw arrays,
per-candidate status, dependencies and audit output are exclusive writes.
One constructed root cannot certify PC0, PC1, PC3, PC4, or task-wide scoring
validity; a negative rank association does not by itself diagnose its cause.
