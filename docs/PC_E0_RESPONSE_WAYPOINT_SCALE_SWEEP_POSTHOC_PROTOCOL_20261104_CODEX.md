# E0 response validity along saved optimized directions

This is a post-hoc, fixed analysis of all 48 previously optimized waypoint
prefixes: eight roots, three direction arms, two branches each. It performs no
planning and cannot establish PC1 or evaluate new search performance.

For each saved float32 prefix `p`, use the same repeated `[0.25, 0]` nominal
`u0` and stored five-step finite-difference response `J`. Evaluate actual
float32 actions `u(alpha)=float32(u0+alpha*(p-u0))` at fixed
`alpha = (0, 0.05, 0.1, 0.25, 0.5, 1.0)`. Compute the linear estimate with
the **actual rounded** `u(alpha)-u0`, not the unrounded proposal. Save
endpoints, absolute error, displacement-normalized error when displacement
is greater than 1e-10, goal-distance change/sign and the maximum normalized
action displacement. Record wall-blocking event changes, including ties and
near-zero denominators without artificial ratios.

At every root calculate the analytic piecewise Jacobian of the E0 unicycle at
the nominal action by propagating derivatives through the recorded fixed
blocking branch. Compare to the saved center-difference Jacobian, and report
whether all 21 actual coordinate probes follow the nominal blocking sequence.
For each nonzero saved action direction, compare its analytic directional
derivative to an independently evaluated central directional difference with
action-space norm `1e-3`, and record the two corresponding branch sequences.
The analytic expression is a derivative only where the selected collision
branch is locally unchanged; do not use it across a discontinuity.

Recompute with the exact backend and independently implemented wall
intersection; bind all dependencies and source audit hashes. Report all
48 directions and all six scales; aggregate by scale and separately flag
branch changes. These samples were selected by prior optimizers and share
roots; they are not independent or representative. No adaptively selected
cutoff or statistically confirmed trust region is authorized.
