# Fixed-prefix cp2 velocity cross-tab

The predeclared diagnostic rescored the same 36 exposed prefixes under three
initial velocity inputs. The causal input came from the two successive
post-step RGB observations already used by the planner. The qvel input came
from the offline position backend and is privileged: it was not used for
action selection or execution and is not certified as the instantaneous
legacy-renderer qvel. All original causal scores reproduced.

| Root | Causal vs actual RGB / physical | Zero vs actual RGB / physical | Offline qvel vs actual RGB / physical |
| ---: | --- | --- | --- |
| 8343 | 3 / 4 | 9 / 4 | 4 / 5 |
| 8345 | 3 / 3 | 6 / 6 | 7 / 7 |
| 8349 | 6 / 6 | 3 / 3 | 3 / 3 |
| 8351 | 6 / 7 | 7 / 8 | 1 / 0 |
| Total | 18 / 20 | 25 / 21 | 15 / 15 |

Numbers are ranking reversals among 36 dependent within-root pairs. Causal
RGB velocity improves aggregate alignment with the actual RGB-derived
score relative to zero, but improves the physical ranking by only one pair
in aggregate and is worse on root 8349. Offline qvel is also not uniformly
better. The task score is based on proprio encoding of RGB-estimated state,
not visual-token error.

Mean absolute score error against the actual RGB-derived score is lower for
causal than zero on roots 8343, 8345 and 8349, but higher on 8351. This does
not measure deployed closed-loop gain. The saved model hash was unchanged;
49 encodes and 108 predictions were used, with no new physical steps.

The fixed bank was generated from the causal state and remains fixed across
modes. Scores are dependent observations from exposed roots, and qvel is
instantaneous whereas the RGB estimate is a finite-interval proxy. The
result supports further interface and planning tests, not a P1/PC pass.

Protocol: `PC_CAUSAL_CP2_VELOCITY_CROSSTAB_PROTOCOL_20261103.md`.
Complete result:
`le-wm-main/results/pc_causal_cp2_velocity_crosstab_20261103_DEV01.json`.
