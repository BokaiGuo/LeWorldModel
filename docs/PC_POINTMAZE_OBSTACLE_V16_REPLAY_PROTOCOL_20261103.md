# Obstacle challenge action-clock and fixed-trace replay

Use the already exposed 24-root equal-stratum V16 obstacle panel, its saved
four-controller traces and the exact extension states. Inspect geometry at
the actual controller start, `states[root, 40]`, as well as frame 0.
Reconstruct each saved action sequence using V16's `_step`, which resets
the simulator from the preceding state and performs five MuJoCo integrator
steps per recorded block. Record initial, block-1, block-5 and terminal
distances to the declared goal; compare reconstructed terminal distances
and block counts to the source report, preserving all 96 rows and errors.

No learned inference, candidate choice, root selection by outcome, or new
controller is performed. Geometric straight-line obstruction alone does
not prove that every successful action sequence has a non-greedy prefix.
The source's one-step oracle is privileged, waypoint-conditioned and not
perfect long-horizon MPC. This exposed panel and its replay cannot establish
Planning Compiler P1, independent generalization, or method superiority.
