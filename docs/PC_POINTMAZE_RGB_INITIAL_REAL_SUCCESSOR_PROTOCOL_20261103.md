# Real-successor versus model score on sealed RGB-position CEM banks

For all 384 previously sealed two-packet candidates in
`a0_pointmaze_rgb_initial_candidate_oracle_20261103_DEV01`, replay the same
reset and action sequence in the legacy native RGB renderer. Save cp10 RGB
and terminal state for every candidate. Require terminal positions to match
the separately sealed position-backend replay exactly; no candidate selection
or filtering is permitted. Renderer observations are post-step without
additional `forward`, as in the source execution bridge.

On host CUDA, encode every cp10 RGB with the same frozen encoder as the
planner and score it against the same real RGB goal using the initial task
mask. Compare three same-candidate orderings per bank: the saved model
predicted latent cost, the actual RGB latent cost, and the prior position
distance. Report all 48 scores, pairwise reversals and selected candidate
indices. These are post-outcome diagnostics: actual cp10 RGB and simulator
state must never enter the deployed selector. Keep exact-tie counts.

Save raw RGB or losslessly compressed arrays, fingerprints, source
identities, physical state alignment and model hashes. Four roots and eight
adaptive banks are exposed development data; 384 candidates and 9,024
within-bank pairs are dependent. Physical metric, RGB latent score and model
predicted score measure different objects; a positive diagnostic does not
establish full planning performance, independence, causation, equal budgets
or P1. Encode only on CUDA; no new training.
