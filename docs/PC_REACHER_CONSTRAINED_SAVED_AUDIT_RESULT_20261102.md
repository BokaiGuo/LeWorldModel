# Constrained search: saved action and goal-Q verification

The CPU saved-data audit checked both roots, all 42 candidate action arrays,
all five common-tail action arrays per root, strict selector flags and
winner ordering. Normalized actions decode exactly to the stored float32 raw
actions. It confirmed `null` selection on both roots. The CPU audit could
not independently recalculate goal-Q scores because the original model-only
report omitted the goal latent; this omission is a reproducibility gap, not
evidence of a numerical error.

A separate host-CUDA replay reconstructed the hash-bound goal images and
frozen checkpoint and encoded each goal once. It saved those newly obtained
goal coordinates and recomputed all 105 endpoint Q scores per root from the
original saved endpoints and fixed Gram. All 210 scores matched the saved
compiler scores exactly in this run (maximum absolute error 0.0 on each
root), and model parameters remained unchanged. This verifies score
reconstruction with the same implementation and frozen model; it does not
independently validate the world model, Q's task meaning or root provenance.
No candidate was reselected and no new physical outcomes were used.

The CPU audit SHA-256 is
`d850ee13a0d6349e9df4b4ca3637e65f9abec4e3580121d33cb7a4a8f871cfab`.
The separate CUDA goal replay is
`le-wm-main/results/reacher_pc_constrained_goal_q_replay_20261102_v1/report.json`,
SHA-256 `5b36c1ac7b35923fcd8f5d28e22340371ab9eb67d96acd72644cb573d8e6f066`.
Both goal coordinates and recomputed scores are saved there.

Eight tests passed, including tampered raw action and selector checks and
analytic batched Q-score checks. This outcome does not establish P1,
physical task success, equal compute, or independence.
