# Privileged physical-CEM MPC and old-tail retention

On the four exposed development roots, a position-backend CEM used the same
two-packet then one-packet search schedule, Gaussian innovations, population
and action bounds as the earlier visual CEM. Its objective and state access
were different: it used simulator state and privileged goal position.

| Root | Physical CEM | Retained-tail physical CEM | Visual random CEM | Visual spectral CEM |
| --- | ---: | ---: | ---: | ---: |
| 1884 | 0.00523 | 0.00644 | 1.02661 | 1.67904 |
| 1885 | 0.27239 | 0.07032 | 0.99383 | 2.40032 |
| 1886 | 0.02792 | 0.02134 | 0.82040 | 0.95427 |
| 1887 | 0.00952 | 0.03603 | 2.01301 | 1.60678 |

Values are position-backend distance to archive frame 10, not native success.
All methods execute 10 high-level actions; physical and visual scores and
computational operations are different, so equal candidate counts do not
imply equal resources. The privileged arm provides a development oracle
comparison, not an RGB-only baseline or proof of perfect dynamics.

On root 1885, the original plan's physically rescored tail cost was 0.07032,
but fresh CEM discarded it and chose a tail costing 0.27239. After observing
this outcome, the development follow-up replaced one of the 48 tail proposals
with the old tail. The resulting 0.07032 illustrates a real finite-sample
replanning failure and an actionable retention control. On roots 1884 and
1887 the warm version was worse than the original cold CEM: using one slot
for retention also changes the elite updates. The guarantee applies only
relative to the retained candidate under the **same physical scoring**, not
to an unknown real outcome or to cold-CEM performance.

Each physical CEM run scored 48 ten-action candidates and 48 five-action
candidates (96 rollouts; 720 high-level simulated steps), plus execution and
verification. The warm follow-up scored 48 five-action tails (240 simulated
steps) and repeated the native RGB execution. Selected chains were executed
through the native RGB bridge; the position backend's split and continuous
four-state outputs agreed within `1.01e-16` on these eight chosen chains.
This does not verify its full-state equivalence to the RGB bridge's hidden
MuJoCo state. Three tests passed, including recomputation of every saved
physical candidate score.

The old-tail control was designed after viewing the initial result and uses
future physical state/goal for scoring; it cannot be applied directly as an
RGB planner. A deployed version must rescore the old tail using the same
frozen-model objective and real RGB history available to new candidates,
include the rescoring in its budget, and be evaluated in a new physical
closed loop. Neither this diagnostic nor these roots establish P1.

- Initial protocol: `docs/PC_POINTMAZE_PHYSICAL_CEM_MPC_PROTOCOL_20261103.md`
- Follow-up protocol: `docs/PC_POINTMAZE_PHYSICAL_CEM_WARM_TAIL_PROTOCOL_20261103.md`
- Physical report SHA-256:
  `7fed2f87bfce3119a28a84ee895a408cdebe3f052a69850919e43a800332cd51`
- Warm-tail report SHA-256:
  `d43a8415d3e597d5c173f5f212916c9eddab1563d45c4f3202f411271e130914`
- Validation: `le-wm-main/tests/test_a0_pointmaze_physical_cem_mpc_20261103.py`
