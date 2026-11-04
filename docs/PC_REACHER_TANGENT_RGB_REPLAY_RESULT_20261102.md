# Tangent action bank: same-action RGB replay

The fixed protocol `PC_REACHER_TANGENT_RGB_REPLAY_PROTOCOL_20261102.md`
completed 70/70 branches on host CUDA, each with checkpoints at physical
steps 5, 10 and 15. Every step-15 open prediction matched the previously
sealed predicted endpoint to the fixed tolerance; every checkpoint qpos
distance matched its saved physical trajectory within 1e-8. The root,
goal, action normalization and model parameter hashes were bound.

Mean latent MSE to the real RGB encoding is shown below. Refreshing means
predicting a single packet from the immediately preceding real RGB and
executed-action history; no refreshed action was selected.

| Root | Step | Open prediction MSE | Refreshed prediction MSE | Refreshed lower |
| --- | ---: | ---: | ---: | ---: |
| 20263160 | 5 | 0.004581 | 0.004581 | 0/35 |
| 20263160 | 10 | 0.007820 | 0.006024 | 34/35 |
| 20263160 | 15 | 0.014171 | 0.007065 | 35/35 |
| 20263161 | 5 | 0.005280 | 0.005280 | 0/35 |
| 20263161 | 10 | 0.011123 | 0.005236 | 35/35 |
| 20263161 | 15 | 0.018913 | 0.006328 | 35/35 |

At step 5 the two predictions are identical by construction. The branch
counts are not independent experiments: they share roots, histories and
actions. Lower MSE does not demonstrate improved closed-loop control.

For each root, compare each of the six non-neutral prefixes against prefix 0
under each of the five identical tails: 30 paired contrasts. At step 15:

| Root | Open-predicted goal-Q vs actual RGB goal-Q reversal | Actual RGB goal-Q vs physical qpos-distance reversal |
| --- | ---: | ---: |
| 20263160 | 1/30 | 14/30 |
| 20263161 | 0/30 | 9/30 |

Here reversal means strictly opposite signs for the corresponding paired
differences; no significance test or independence is implied. For the one
open/actual reversal the absolute Q margins are at least 0.001776 and
0.001046 respectively. For actual-Q/qpos reversals the smallest absolute
margins are 0.001305 Q and 0.000306 qpos on the first root, and 0.000596 Q
and 0.002166 qpos on the second. Q and qpos are distinct metrics and units.
These results support investigating evaluation alignment as well as
prediction drift; they do not prove either is the sole cause of failure.

Predictions, observed latent vectors, hashes and Q values are retained in
`le-wm-main/results/reacher_pc_tangent_rgb_replay_20261102_v1/report.json`,
SHA-256 `4a0c2774ee5569476c668301a53ba8ff6ea183b6a1c3ed1255637f2a65d0f27e`.
Nominally each completed branch used three open and three refreshed token
prediction samples; the report does not give full encoder, inherited goal-Q,
or wall-time costs. No physical success, equal-budget control advantage,
independent generalization or P1 admission is established.
