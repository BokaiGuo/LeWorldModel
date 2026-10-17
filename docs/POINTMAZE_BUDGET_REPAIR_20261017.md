# PointMaze online repair: planner-budget diagnosis

## Finding

The earlier 24-block PointMaze panel was not a clean test of action-effect
repair. On the same opened roots (1900--1931), the RGB visual-dynamics policy
and the privileged one-step candidate oracle both reached 23/32, while the
effect policy with HTSC reached 24/32. Thus the oracle and the visual policy
shared the same ceiling, and the remaining failures could not be attributed to
the frozen latent alone.

Increasing only the closed-loop execution budget to 40 blocks changed the
privileged exact-dynamics expanded-bank ceiling from 23/32 to 32/32, with
mean final distance 0.1141. This identifies insufficient execution budget and
long-horizon reachability as the dominant cause of the earlier negative result.

With the same 40-block budget and RGB-derived inputs on roots 1900--1931:

| policy | success | mean final distance |
|---|---:|---:|
| visual dynamics | 32/32 | 0.1586 |
| effect adapter | 32/32 | 0.1598 |
| effect adapter + HTSC | 32/32 | 0.1588 |

On an unopened root panel (1932--1963), using calibration roots 1800--1849
only:

| policy | success | mean final distance |
|---|---:|---:|
| visual dynamics | 32/32 | 0.1411 |
| effect adapter | 32/32 | 0.1536 |
| effect adapter + HTSC | 32/32 | 0.1613 |

## Interpretation boundary

The budget repair removes the prior success-rate failure, but it does not
show that HTSC improves over the strong visual-dynamics baseline. The
independent panel instead shows equal success with lower distance for visual
dynamics. The effect adapter therefore remains a local action-effect repair,
not a universal planning improvement.

The runtime-pulse retrained adapter was also tested separately and performed
worse online than the paired-train adapter, despite better offline effect
regret. Offline effect accuracy is therefore insufficient evidence for RGB
closed-loop transfer.

All privileged oracle results use simulator state only for evaluation and
candidate scoring. They are ceilings, not deployable RGB policies.
