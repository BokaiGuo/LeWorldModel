# Fixed goal-encoder co-metric on sealed Reacher trajectories

Use the two known roots and every saved arm/checkpoint from
`reacher_pc_feasible_prediction_replay_20261102_v1/report.json`. Reconstruct
their goal RGB and history from the bound root source and fixed seed. Check
history index, root and goal RGB hashes, model weights/stats hashes and frozen
parameter hash. Compute the existing goal-only encoder co-metric
`Q=.01*(J_E J_E^T+.01 I)^-1` with its unchanged CUDA full Jacobian and
finite-difference checks. This uses no physical future or labels.

Apply the same root-specific Q to the saved open predictions, refreshed
predictions, and actual RGB-encoded latents at steps 5/10/15. Also reconstruct
the original Euclidean goal costs and compare against the saved replay before
accepting the result. Save goal encoding, Q's Gram, all gradient checks, scores,
and exact input/source hashes. Separately report score pair-order comparison
with saved qpos distance. This is retrospective score evaluation on exposed
development trajectories, not a new selector, physical episode, equal-budget
experiment, or P1 confirmation. A favorable score would require a newly frozen
selector and independent physical validation before any method claim.
