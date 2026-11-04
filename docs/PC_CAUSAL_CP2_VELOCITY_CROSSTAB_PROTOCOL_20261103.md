# Causal prelude, zero and privileged velocity cross-tab

Reuse the 36 already-exposed fixed action prefixes from the
`pc_causal_history_full_bank_replan_20261103_DEV01` receipt. The prelude,
candidate actions, goal, model and outcomes are fixed. Re-evaluate each
five-action prefix from the identical saved cp2 RGB with three cp2 proprio
inputs: the recorded RGB two-post-step velocity estimate, the same
RGB-estimated position with velocity zero, and the same RGB-estimated
position with the saved position-backend cp2 qvel. The last is a privileged
offline diagnostic only and must never enter the RGB-only planner; this
position backend is audited for position, not independently certified for
instantaneous qvel equivalence to the rendered legacy simulator.

Encode each actual cp7 RGB with the saved cp6/cp7 causal frame difference.
Predict using the same frozen checkpoint, original action normalization and
affine position score; compare causal-mode scores to the original saved
first-packet scores before using physical endpoints. Save all scores,
per-root pairwise ordering reversals/ties against actual RGB-derived scores
and physical position distance, source/model hashes and actual call counts.
Do not generate, alter or select actions, or re-run physics. Evaluating
previously exposed candidates is a diagnosis, not independent confirmation.
The RGB frame velocity is a finite-interval proxy and the privileged qvel is
instantaneous; neither rendering time alignment nor generalization is assumed.
