# V21: Fresh-root C3 calibrated decision reliability

V21 evaluates C3 without reusing V20's fit, calibration, or outcome roots.
It freezes the V20 recurrent/action model exactly as recorded in its sealed
selection artifact (including its declared history source) and its causal
visual-motion `motion >= 0.02` route, then creates fresh mixed-velocity PointMaze roots
`28040..28871`.

The 256/256/256 train/calibration/outcome partition is root-disjoint. Only
calibration roots may set the rootwise contrast envelope and model-only risk
threshold. The final outcomes are materialized only after
`v21_frozen_c3_gate.json` is written.

The endpoint is fixed-bank held-out regret/coverage against action-only
matched fallback. Any coverage statement is marginal under exchangeable roots,
not permit-conditional, adaptive-CEM safety, online MPC, contact transfer, or
a universal world-model claim. A zero-release or non-improving risk-coverage
result is a C3 rejection, not a retuning opportunity.
