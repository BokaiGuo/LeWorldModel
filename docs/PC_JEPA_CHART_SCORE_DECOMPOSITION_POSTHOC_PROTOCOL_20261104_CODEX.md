# JEPA two-axis chart: saved-bank score decomposition

This post-hoc diagnostic uses only the already sealed 64-chain v12 contact
score bank and neural output; it does not execute the model, environment,
planner, or expose the privileged witness variants to the planner.

The worker uses an orthonormal two-axis chart scaled by `1/sqrt(D)`,
projects `(predicted-root)`, and scores `mean((predicted-goal)^2)`.
Thus for each saved chain, squared distance from its projected position to
the saved chart goal is the score contribution in those two directions;
full MSE minus this contribution is the orthogonal residual, subject to
floating-point error. Report negative residuals before rounding, projection
fractions, and pairwise ordering agreement (separately excluding ties
within `1e-8`) between projected distance and full MSE. Report separately
the 56 planner-sampled chains and 8 privileged witness variants; they are
highly correlated, not independent observations. Also compare each score
to the already saved actual-terminal-encoder score where appropriate.

Bind the source audit, report, frozen neural decision, and source code via
hashes, validate shapes and equality with saved report values, and save a
new non-overwriting artifact. The decomposition assumes the worker's
declared chart construction and cannot reconstruct omitted coordinates,
prove chart orthonormality numerically from unsaved bases, establish a
physical control failure cause, or certify PC1/PC3. No extrapolation to
other roots, models or contact states.
