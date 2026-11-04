# Lookahead: same-action prediction versus real RGB and physics

The [fixed diagnostic](PC_POINTMAZE_LOOKAHEAD_SAME_ACTION_PROTOCOL_20261103.md)
executed all 36 sealed prefix/optimized-suffix action chains across four
exposed roots. All step-five RGB frames exactly matched the sealed bank.
The 36 open-loop predicted target costs matched the source lookahead scores
within the fixed tolerance. The maximum normalized action round-trip
discrepancy was `5.97e-8`; this checks this specific action path, not every
possible interface or initial-velocity assumption.

Each row below has 36 comparable within-root pairs; reversal counts compare
the stated ordering at the same ten-action time:

| Root | Open prediction vs actual encoding | RGB-refresh prediction vs actual encoding | Actual encoding vs physical distance |
| --- | ---: | ---: | ---: |
| 8343 | 6/36 | 6/36 | 6/36 |
| 8345 | 4/36 | 2/36 | 3/36 |
| 8349 | 17/36 | 2/36 | 7/36 |
| 8351 | 3/36 | 4/36 | 0/36 |

The proprio-token mean squared error against the actual encoded token was
lower with refreshed context on all four roots: open/refresh respectively
`0.002852/0.000335`, `0.000968/0.000300`, `0.002396/0.000264`, and
`0.000184/0.000140`. Refresh improved pair ordering on two roots, tied on
one and worsened on one; reduced MSE does not guarantee better control.

The actual-token calculation uses RGB-derived position and finite-difference
velocity, not privileged simulator state, so its error includes the RGB
state estimator. Initial velocity is assumed zero. Physical distances come
from a separate endpoint replay; no ground-truth simulator state was used
to select actions. Pairs share four roots and are not independent samples.
The results support separating rollout drift and task metric alignment,
but cannot establish a unique failure cause or P1 passage.

Receipt [DEV01](../le-wm-main/results/pc_pointmaze_lookahead_same_action_20261103_DEV01.json)
SHA-256 `d9dde1cc4c341e4a8b2907436266d3b162a803cb75fc49f8b190d75d02d4dc46`.
