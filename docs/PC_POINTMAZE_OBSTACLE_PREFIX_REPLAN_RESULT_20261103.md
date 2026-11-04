# Conditional RGB feedback planning after obstacle prefixes

All 12 runs in the [fixed protocol](PC_POINTMAZE_OBSTACLE_PREFIX_REPLAN_PROTOCOL_20261103.md)
completed on four exposed roots. Every trajectory used its own real
step-5 RGB and then the same random-initialized CEM and four feedback
decisions, with 50 scored candidates per decision. All terminal frames
matched a fresh bridge replay. The four random-prefix chains exactly
reproduced their full source actions and RGB frames.

| Root | Random prefix endpoint distance | Spectral prefix endpoint distance | Zero prefix endpoint distance |
| --- | ---: | ---: | ---: |
| 8343 | 0.031564 | 0.091081 | 0.244271 |
| 8345 | 0.334538 | 0.374627 | 0.504585 |
| 8349 | 0.049455 | 0.339885 | 0.015041 |
| 8351 | 0.412133 | 0.377970 | 0.606551 |

Spectral is closer than random on 1/4 roots and closer than zero on 3/4.
Using the predeclared offline endpoint threshold `<=0.25`, random and zero
each meet it on 2/4 roots, spectral on 1/4. This is not native online
success. The total source-prefix computation was not equalized, and the
results are conditional on one common optimizer and exposed roots.

All four spectral prefixes had already improved goal distance at step 5.
Thus even a favorable endpoint would not prove the requested non-immediate
progress mechanism. This negative conditional comparison does not prove a
theoretical impossibility: the candidate, scoring, observation and search
contracts remain limited. No independent generalization or strong-baseline
advantage is established.

Receipt [DEV01](../le-wm-main/results/pc_pointmaze_obstacle_prefix_replan_20261103_DEV01.json)
SHA-256 `35e8c18236849675a8e6a6caafa584a82ab3d9e52ba629cde87e22a929256b2b`.
