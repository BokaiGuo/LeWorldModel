# PointMaze obstacle RGB spectral-CEM: development result

The [fixed protocol](PC_POINTMAZE_OBSTACLE_RGB_SPECTRAL_DEV_PROTOCOL_20261103.md)
completed all eight trajectories: four already-exposed roots, two arms,
25 high-level physical actions each, and five real-RGB decisions per arm.
All eight terminal images matched a fresh bridge replay; endpoint positions
were obtained from the audited position replay backend. The bridge does not
expose its own state, so the two endpoints were not independently compared.

| Root | Random endpoint distance | Spectral endpoint distance |
| --- | ---: | ---: |
| 8343 | 0.031564 | 0.116117 |
| 8345 | 0.334538 | 0.391371 |
| 8349 | 0.049455 | 0.056398 |
| 8351 | 0.412133 | 0.179328 |

Spectral ends closer on one of four roots. At the predeclared offline
endpoint threshold `<=0.25`, random meets it on 2/4 roots and spectral on
3/4. These are not authenticated online success rates and the small,
previously exposed panel cannot establish an advantage. Per arm, each of
20 decisions scored 50 candidates; spectral additionally used 40 JVPs and
20 SVDs. Equal prediction counts are not equal total budget.

The planner used RGB-derived state and a rendered goal. Reset state and goal
coordinates were used to initialize the environment, render the goal, and
evaluate outcomes, not to select actions. The archived roots, normalizer
and checkpoint training provenance are not independently certified. No
common-suffix intervention, equally funded strong baseline, independent
confirmation or reconfiguration mechanism is established; neither this
result nor the prior negative results alone determine P1.

Receipt: [DEV01](../le-wm-main/results/pc_pointmaze_obstacle_rgb_spectral_dev_20261103_DEV01.json),
SHA-256 `e5955123b5b88797a66e4c703a1c0f5b321a23c8fbead371dc029c8d44a5ff91`.
Saved-data tests do not rerun inference or physics.
