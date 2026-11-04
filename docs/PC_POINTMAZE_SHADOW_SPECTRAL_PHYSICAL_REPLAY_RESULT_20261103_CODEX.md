# Physical replay of the paired spectral shadow development run

The four saved 17-action chains were replayed without changing actions. The
same shadow bridge reproduced all 18/18 RGB frames for each chain. An
evaluator-only MuJoCo 2.1.2 replay and the established MuJoCo 2.2 position
backend produced exactly equal final position coordinates for all four chains.
The following are replayed Euclidean position distances to the fixed goal;
lower is better:

| Exposed root | Arm | Shared step-12 distance | Step-17 distance | Wrapper `<0.5` at endpoint | Stop |
| --- | --- | ---: | ---: | :---: | --- |
| r8343_a0_p2 | baseline | 1.260430 | 0.762595 | no | scheduled horizon, 17 actions |
| r8343_a0_p2 | spectral | 1.260430 | 0.705294 | no | scheduled horizon, 17 actions |
| r8345_a0_p2 | baseline | 2.131309 | 1.708919 | no | scheduled horizon, 17 actions |
| r8345_a0_p2 | spectral | 2.131309 | 1.603303 | no | scheduled horizon, 17 actions |

The `<0.5` position threshold comes from the existing PointMaze wrapper and
the earlier cp12 planning implementation contract; it was **not specified as
a primary endpoint in this paired run's protocol** and is reported here as a
post-hoc wrapper diagnostic, not native success. All four replayed paths stay
outside that threshold at every recorded high-level step. The shadow bridge
did not implement a native success/early-stop condition; all four completed
the scheduled horizon, with no recorded dispatch failure or censoring.

The physical replay therefore agrees with the direction of the earlier
RGB-proxy comparison on these two roots. The differences are approximately
0.0573 and 0.1056 distance units. These values are from deterministic replay,
not privileged-state logging during the original run. Frame equality and
cross-version endpoint agreement are useful checks but cannot by themselves
prove identical hidden state in the original simulator.

Both methods start from the **same previously executed prefix**. This tests
spectral initialization of short-horizon replanning, not whether a
spectral-generated non-greedy prefix changes subsequent control capability.
There are two already-exposed roots, only five feedback decisions, no native
success measurement, and the spectral arm has additional Jacobian/SVD cost.
There is no equal-total-resource random-initialization or strong-planner
comparison, and no independent confirmation. This is not P1 admission.

The [receipt](../le-wm-main/results/pc_shadow_spectral_physical_replay_20261103_CODEX.json)
contains all evaluator states, input hashes, distances and cross-backend
discrepancies. The
[protocol](PC_POINTMAZE_SHADOW_SPECTRAL_PHYSICAL_REPLAY_PROTOCOL_20261103_CODEX.md)
was fixed before replay; evaluator state was not supplied to either planner.
