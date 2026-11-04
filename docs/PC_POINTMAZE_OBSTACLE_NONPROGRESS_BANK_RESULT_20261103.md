# Model-selected non-progress spectral prefixes: development result

The [protocol](PC_POINTMAZE_OBSTACLE_NONPROGRESS_BANK_PROTOCOL_20261103.md)
executed all nine fixed candidates on each of four exposed roots: 36 real
five-action physical trajectories. Selection used the frozen model before
physical outcomes. Every root had an eligible nonzero predicted non-progress
candidate, and all four selected candidates also had no immediate reduction
in physical goal distance:

| Root | Selected bank index | Initial distance | Selected step-5 distance | Zero step-5 distance |
| --- | ---: | ---: | ---: | ---: |
| 8343 | 6 | 1.373291 | 1.487012 | 1.433311 |
| 8345 | 6 | 2.048280 | 2.258941 | 2.051080 |
| 8349 | 4 | 1.591247 | 1.604257 | 1.650658 |
| 8351 | 4 | 2.627013 | 2.706745 | 2.566226 |

These are not independent roots. The selected action on root 8349 still
does better than zero at step 5; the condition here compares with the
initial distance, not with zero. Initial RGB velocity was assumed zero,
and the model score is in learned proprio space, so agreement on these
four selected cases does not establish general prediction calibration.
Candidate choice used no future-value or reachability score and all nine
arms were executed for diagnostic coverage. No matched future-action
intervention or reoptimized continuation has yet been performed on this
new bank; neither useful reconfiguration nor P1 follows from this result.

The historical AP-compatible CEM updates unprojected elite proposals after
scoring projected ones; local inspection confirms this matches the checked
official AP source, so it is not evidence of a local implementation mismatch.

Receipt [DEV01](../le-wm-main/results/pc_pointmaze_obstacle_nonprogress_bank_20261103_DEV01.json)
SHA-256 `e45ac931ba3da9c480cc37c125c00d202ee7dd4014abba71ffd88c238b928e1b`.
