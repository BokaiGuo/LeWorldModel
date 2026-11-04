# Five-action task-response bank: exposed-root development

The fixed protocol evaluated nine full five-action programs on each of four
previously exposed PointMaze obstacle roots, using the frozen model and real
RGB simulator observations. The task projection uses a proprio representation
from RGB-estimated state and an analytic affine inverse; it is not a pure
visual-latent distance. The 16 nullspace programs use two particular SVD
basis vectors and their signs, not the entire eight-dimensional nullspace.

| Root | JVP vs FD relative error | Model score vs RGB-derived score reversals | Model score vs physical distance reversals | RGB-derived score vs physical distance reversals |
| ---: | ---: | ---: | ---: | ---: |
| 8343 | 0.001112 | 5 / 36 | 6 / 36 | 1 / 36 |
| 8345 | 0.000960 | 7 / 36 | 6 / 36 | 1 / 36 |
| 8349 | 0.001564 | 8 / 36 | 7 / 36 | 1 / 36 |
| 8351 | 0.001248 | 2 / 36 | 2 / 36 | 0 / 36 |

Pairs share roots and actions; 144 pairwise comparisons do not represent 144
independent tasks. The finite-difference check uses a small 0.001 step and
verifies this numerical response implementation, not prediction correctness
or linearity at the 0.5 candidate amplitude.

The selected null-basis actions had essentially zero linear task response by
construction; their mean nonlinear *model* position-response norm at the
actual candidate amplitude was 0.00190, versus 0.26400 for the four
nonzero-spectral actions per root. Mean actual physical displacement relative
to the same root's zero program was:

| Root | Spectral actions | Null-basis actions | Largest null-basis action |
| ---: | ---: | ---: | ---: |
| 8343 | 0.25430 | 0.00648 | 0.00755 |
| 8345 | 0.17775 | 0.03189 | 0.11381 |
| 8349 | 0.20728 | 0.00614 | 0.01439 |
| 8351 | 0.23304 | 0.01659 | 0.03271 |

These are differences from zero-program cp5 position, not task improvement
or proof of a changed future control capability. One null-basis action at root
8345 moved the physical position by 0.11381 relative to zero despite its
small predicted task response. Initial RGB velocity is assumed zero, whereas
the physical reset can have nonzero velocity; contact, image-state estimation,
model prediction and action-coordinate effects are not separated here.
Eleven of 16 null-basis programs did not improve physical distance relative
to the initial state, but this is only an immediate outcome.

The run saved all actions, six-frame histories, JVP/FD matrices, basis vectors,
physical endpoints and individual model outputs in the bound JSON/NPZ.
It recorded 40 JVP calls, 80 finite-difference samples, 36 candidate
predictions, 41 encodes, 180 RGB-simulator steps and 180 position-backend
replay steps. JVP internals, wall time and all end-to-end costs are not
certified against another planner. A matching suffix intervention and
actual per-prefix replanning are still required; no P1/PC gate is opened.

Protocol: `PC_POINTMAZE_FIVE_ACTION_RESPONSE_BANK_PROTOCOL_20261103.md`.
Report: `le-wm-main/results/pc_pointmaze_five_action_response_bank_20261103_DEV01.json`.
