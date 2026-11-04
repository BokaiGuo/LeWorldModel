# v17 score-augmented response: E0 multistart baseline development

This new eight-task exact-Wall panel tests planning quality and allocation,
not the previously stopped v17 mechanism panel. Tasks: start
`[-0.5,y,theta]`, `y` in `{-0.29,0.29}`, `theta` in `{-1.3,1.3}`, goal
`[0.5,gy]`, `gy` in `{-0.29,0.29}`. Check exact task tuples against all
enumerated prior v5–v17 and fixed-bank panels; wider provenance is unknown.

Three arms use the same task, legal actions, 15-step horizon, deterministic
global proposal bank and a cap of 4,095 five-step prediction tokens:

* `v13`: unchanged chart-only response compiler.
* `v17`: score-augmented compiler with score scale 1 and target 0.
* `multistart_cem`: four best distinct *indices* from the same scored global
  bank, each initialized at its bank chain, with the remaining full-chain
  proposals distributed as evenly as possible; 15-knot per-step CEM,
  independent deterministic streams and final incumbent/zero-prefix
  pair. Identical action values from different indices are allowed and
  reported. Every bank score, sample, final comparison and cost is charged.

Record actual predictions, complete-chain incumbents, terminal distance,
success `<0.05`, SVD calls, and synchronized CPU wall time. Equal prediction
tokens are not an equal total hardware budget; do not infer cost advantage.
Audit all saved actions and endpoints against alternate same-environment
geometry; check common bank, per-start counts and final best complete-chain
selection. Preserve failures in the denominator.

Stop after eight tasks; no retries, root replacement or post-outcome tuning.
This is a bounded development comparison, not an independently sourced
PC3 confirmation, an upper bound on CEM, or a PC2 mechanism intervention.
Do not promote method or baseline based on an isolated positive root.
