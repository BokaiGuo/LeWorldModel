# Fixed obstacle-chain decomposition: development result

The predeclared diagnostic replayed all 36 previously exposed ten-step
chains on four obstacle roots. Each reset and cp5 image matched its sealed
source, the separate position backend reproduced cp10, and all 36 open-loop
model scores reproduced the original saved score within the stipulated
tolerance. Frozen model parameter hashes matched before and after.

The task score here is the affine inverse applied to the **proprio
representation encoded from a state estimated from RGB**, not a visual
latent distance. The initial state is estimated from one image; cp5 and cp10
velocities use the preceding pair of RGB frames. This causal image-derived
velocity is a time-averaged estimate rather than certified instantaneous
velocity. The simulator state was used only for reset and offline position
evaluation, not for model input or action selection.

Within each root the nine candidates give 36 dependent pairs. Across four
roots:

| Comparison | Ranking reversals / 144 pairs |
| --- | ---: |
| Open predicted score vs actual RGB-derived score | 13 |
| Refreshed predicted score vs actual RGB-derived score | 14 |
| Open predicted score vs physical distance | 19 |
| Refreshed predicted score vs physical distance | 14 |
| Actual RGB-derived score vs physical distance | 6 |

The open/refreshed proprio latent MSE means are 0.001265 and 0.000340;
refresh reduces MSE for 28 of 36 chains. These are dependent, selected
development actions: the aggregate improvement does not imply that every
ranking improves or that feedback improves closed-loop control.

On root 8349, the original nonprogress example is informative:

| Bank index | Open score | Refresh score | Actual RGB-derived score | Physical distance |
| ---: | ---: | ---: | ---: | ---: |
| zero (0) | 1.3126 | 1.2878 | 1.3146 | 1.2931 |
| nonprogress (4) | 1.2616 | 1.2017 | 1.2828 | 1.2606 |
| model greedy (7) | 0.7391 | 1.2756 | 1.3139 | 1.3142 |

The open model strongly overestimates the greedy chain's benefit. For this
fixed comparison, refresh restores index 4 ahead of greedy and zero, but
actual RGB-derived score still ranks greedy narrowly ahead of zero while
physical distance reverses them. Neither index 4 nor a refreshed plan was
selected or deployed as a result of this analysis.

The run performed 109 encodes, 36 two-packet and 36 one-packet predictions,
360 RGB-environment steps, and 360 independent position-backend replay
steps. Full retained frames and actions, model hashes, per-action metrics,
counts and ranking tables are in
`le-wm-main/results/pc_pointmaze_obstacle_fixed_chain_decomp_20261103_DEV01.json`
and its bound NPZ. Source-code/dependency closure and training-set provenance
are not fully certified. This is not an independent experiment, not a
matched-compute planner comparison, and does not open a P1 or PC gate.
