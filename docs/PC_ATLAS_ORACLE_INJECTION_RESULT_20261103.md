# Strong-action injection into Atlas: development diagnosis

The eight previously exposed unicycle goals were evaluated under the
predeclared injection protocol. For each, the existing strong gradient
planner's complete two-action chain was inserted into the Atlas candidate
bank, and the second action was available at its exact predicted successor.
Both split and direct replays produced identical full-context hashes.

| Goal index | Prefix model cost | Supplied chain terminal model cost | Branch outcome | Final selection |
| ---: | ---: | ---: | --- | --- |
| 0 | 0.0845025 | 1.24e-17 | retained | injected |
| 1 | 0.0974638 | 1.09e-10 | retained | injected |
| 2 | 0.114543 | 2.28e-11 | retained | injected |
| 3 | 0.132393 | 3.69e-15 | retained | injected |
| 4 | 0.154139 | 3.32e-16 | retained | injected |
| 5 | 0.177259 | 1.15e-12 | retained | injected |
| 6 | 0.203616 | 2.63e-15 | retained | injected |
| 7 | 0.229737 | 2.04e-13 | retained | injected |

All eight injected chains were retained and selected; none was scored worse
than the selected chain or discarded in favor of a more costly suffix. The
root cost is 1/3 under the position-only weighted mean over three coordinates.
This is consistent with a candidate-generation or search-coverage limitation
in this particular deterministic exact-dynamics setting. It does not prove
that Atlas scoring, intermediate pruning, or action binding are correct for
other horizons, branches, visual backends or real environments. The prefix
is not a nonprogress witness: every prefix above improves that model metric.

The injected solutions come from another optimizer and are not Atlas
performance. The injection adds a branch and incurs 90 rollout and 8 Jacobian
calls per root, versus 76 and 7 for the old smoke. Three extra diagnostic
replays per root and the gradient planner's own work are excluded from those
compiler counts. This is not an equal-budget comparison.

Artifact:
`le-wm-main/results/pc_atlas_oracle_injection_20261103.json`. The new test
and existing local compiler/atlas tests report 16 passed. None of these facts
opens a scientific gate.
