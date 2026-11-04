# PointMaze registry-normalization sensitivity result

The fixed development protocol is
`PC_POINTMAZE_REGISTRY_NORMALIZATION_PROTOCOL_20261102.md`. The frozen
PointMaze model scored the same 64 action pairs on each of four known roots
on CUDA, using the checked-in JEPA-WM registry's PointMaze statistics.
All model parameters had identical hashes before and after the run. A
serial-versus-batched score check passed at each root.

| Root | Selected pair | Physical regret |
| --- | --- | ---: |
| 1880 | (2, 3) | 0.636081 |
| 1881 | (0, 0) | 0.192195 |
| 1882 | (3, 3) | 0.284369 |
| 1883 | (2, 0) | 0.065061 |

All 256 saved scores are exactly equal to the previous constant-proprio
development run, and the chosen pairs and replay-derived physical regrets
are unchanged. The previous fixed proprio constant exactly equals the
registry mean. This is consistent with the same effective inputs; it does
not establish that normalization has no effect under other statistics.

The checked-in registry says its constants were calculated on training
datasets. That statement does not establish the exact identity of the local
checkpoint's training dataset or rule out overlap with these four roots.
The previous dataset loader calculates statistics from all 2,000 loaded
episodes, including these roots. Identical scores therefore cannot promote
the old result to independent or training-disjoint evidence.

The run performed 256 candidate scores and four additional serial check
scores, plus eight image-encoding calls; elapsed wall time in its report
includes loading and setup. It used previously recorded position-only
physical outcomes and made no new physical interventions. It does not
establish RGB-only online planning, a reconfiguration mechanism, P1, or
generalization.

Result:
`le-wm-main/results/a0_pointmaze_registry_normalization_20261102_DEV01.json`.
