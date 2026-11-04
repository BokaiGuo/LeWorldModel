# Rotated finite-horizon basis diagnostic

Development-only exact unicycle diagnostic on previously exposed goals 0–7,
seeds 101, 202, 303, 404. No independent performance or gate claim.

At each root linearize the position-weighted two-action response at the
fixed feasible reference `[[0, 0.4], [0.6, 0]]`. All arms compute the same
Jacobian and full four-dimensional SVD, including raw controls, so the backend
call budget matches. Generate 16 shared uniform four-ball samples of radius
0.35. Compare raw coordinates, full orthogonal spectral coordinates, and
spectral coordinates with the last two coefficients multiplied by 0.25.
All candidates are offsets from the reference, and must satisfy action bounds
without clipping. Include the same two-step zero and greedy prefixes in each
bank. Optimize one suffix action per prefix with the same local compiler;
select by nonlinear complete-chain task cost and verify split/direct replay.

Record all actions, costs, singular values, basis, backend call counts, wall
times and source hashes. Report root means and paired outcomes without
treating seeds as independent roots. The full isotropic transformation
preserves the sampling distribution but not individual samples; an observed
difference on this finite panel is not evidence of a general spectral
advantage. This protocol changes the action horizon and reference from the
previous zero-point experiment and is not an equal-budget comparison to it.
