# Frozen RGB-context spectral CEM diagnostic

Development-only offline replay of decision 0 for both branches in
`pc_pointmaze_shadow_closed_loop_dev_20261103_CODEX.json`. Verify each saved
decision hash, model parameter hash and action/proprio normalizers before
using its RGB pair and CEM noise. No physical outcomes or simulator state are
inputs to the planner.

Reconstruct the RGB-derived context and goal as in the original executor.
Replay historical CEM and require the selected action and final scores to
match the saved decision within floating-point tolerance. Then inject the
four spectral seeds into the otherwise identical CEM and save its scores,
actions, Jacobian and singular values. Measure the raw-action Jacobian with
central finite differences at epsilon 0.01 for all ten coordinates. Report
absolute and relative errors, not a post-hoc pass threshold.

Count model samples and batches for each CEM arm separately, as well as the
response forward, two reverse vectors, SVD, twenty finite-difference model
forwards and encoding calls. Synchronize CUDA for timing. The FD calls are
diagnostic overhead; the spectral arm itself has extra structure costs.
This is neither equal-total-compute nor real execution, independent evidence,
mechanism validation, or a P1 admission.
