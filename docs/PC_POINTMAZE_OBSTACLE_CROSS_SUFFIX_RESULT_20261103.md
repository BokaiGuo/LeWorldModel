# Obstacle cross-suffix physical intervention

The [protocol](PC_POINTMAZE_OBSTACLE_CROSS_SUFFIX_PROTOCOL_20261103.md)
evaluated all 36 fixed combinations, with 5-action prefixes and 20-action
common suffixes on four previously exposed roots. All eight original chains
reproduced both their sealed final RGB and their recorded position-backend
endpoint. The 36 combinations are physical evaluations, not independent
roots, and no new neural inference was performed.

| Root | Initial distance | Spectral prefix distance at step 5 | Random prefix distance at step 5 |
| --- | ---: | ---: | ---: |
| 8343 | 1.3733 | 1.0499 | 1.2509 |
| 8345 | 2.0483 | 1.9603 | 1.9297 |
| 8349 | 1.5912 | 1.4423 | 1.4962 |
| 8351 | 2.6270 | 2.4080 | 2.3578 |

All four spectral prefixes had immediate target-distance improvement, so
none satisfies a no-immediate-distance-improvement condition. The suffix
response depends on the preceding actions: for example, on root 8349 the
random suffix changes terminal distance relative to the same-prefix zero
suffix by -0.9646 after the random prefix but +0.8166 after the spectral
prefix. This is a useful physical interaction diagnostic, not proof of
better controllability: the prefixes differ in position and velocity, and
the two nonzero suffixes were optimized for their own parent trajectories.
The source trajectories also use successive RGB feedback, whereas these
crossed suffixes are replayed open-loop.

There is no zero/greedy same-budget independent suffix reoptimization,
unexposed-root confirmation, or fair strong-baseline comparison here.
The result cannot establish or reject the full P1 mechanism.

Receipt [DEV01](../le-wm-main/results/pc_pointmaze_obstacle_cross_suffix_20261103_DEV01.json)
SHA-256 `8290574410105d80b5b06dc9782121acf9ab60f97ef621fd056086b53945fa14`.
