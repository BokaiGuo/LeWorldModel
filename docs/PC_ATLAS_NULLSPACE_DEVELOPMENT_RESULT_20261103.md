# Nullspace candidate development results

Two experiments used exact unicycle dynamics and the same eight previously
exposed angular goals; neither is an independent confirmation.

## Full search with additional nullspace branches

The predeclared v3 experiment appended signed task-response nullspace
directions to the original v2 bank. Mean terminal position error fell from
0.141953 for the saved v2 run to 0.112488 for v3. Three of eight v3 selected
prefixes were nullspace directions; five roots had a model-only qualified
reconfiguration branch. The four-start strong gradient comparison had mean
error 0.00000365. V3 used 116 rollout and 18 Jacobian calls per root versus
76 and 7 for v2; this is not a matched-budget performance comparison.

## Four-prefix matched-call ablation

The separate frozen matched-call protocol compared two signed 0.5-scale
nullspace actions against the saved random, nonzero spectral and ordinary
turning candidates. All arms shared zero/progress candidates, the same
one-packet suffix optimizer, four-prefix count and replay. Each arm used 28
rollouts and five Jacobian calls. The shared candidates matched exactly in
all 144 cross-arm comparisons.

| Extra candidate type | Mean terminal position error | Selected extra candidates |
| --- | ---: | ---: |
| Random | 0.304264 | 8 / 24 |
| Nonzero spectral | 0.356722 | 6 / 24 |
| Nullspace | 0.124602 | 15 / 24 |
| Ordinary turns | 0.009901 | 24 / 24 |

Against random, the nullspace arm had 14 wins, 7 ties and 3 losses; against
nonzero spectral, 15 wins, 3 ties and 6 losses; against ordinary turns, it
lost all 24 paired goal/seed cases. There are only eight distinct goals:
the three seeds change the random arm but do not create independent tasks.

Nullspace actions improve coverage relative to these particular spectral and
random proposals, but still lose to the particular ordinary turn proposals.
Their action amplitudes differ, and the extra nullspace SVD and general wall
time were not matched. These findings motivate action-amplitude and optimizer
coverage checks, not a claim that spectral structure has an equal-compute
advantage or that a learned visual model has a physical mechanism.

Results:
`le-wm-main/results/pc_atlas_nullspace_unicycle_dev_20261103.json` and
`le-wm-main/results/pc_atlas_matched_nullspace_dev_20261103.json`.
Protocols: `PC_ATLAS_NULLSPACE_DEV_PROTOCOL_20261103.md` and
`PC_ATLAS_MATCHED_NULLSPACE_DEV_PROTOCOL_20261103.md`.
No existing gate or historical result was modified.
