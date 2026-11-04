# Zero-prefix real-RGB replanning comparison

The [fixed protocol](PC_POINTMAZE_ZERO_PREFIX_PROPRIO_REPLAN_PROTOCOL_20261103.md)
ran eight new ten-step physical trajectories on exposed development roots.
After five zero actions, the planner used its own real cp4/cp5 RGB and
48 predicted-proprio-scored CEM samples to choose and execute a tail.
The frozen model hash was unchanged. Each cp5 physical state exactly
matched the zero-prefix common-suffix panel; all final RGB and positions
matched separate legacy-render and position-backend replay.

| Root | Zero prefix | Random prefix | Spectral prefix |
| ---: | ---: | ---: | ---: |
| 1888 | 0.0346 | 0.0646 | 0.0646 |
| 1889 | 0.6325 | 0.5039 | 0.5039 |
| 1890 | 0.2246 | 0.5669 | 0.5426 |
| 1891 | 0.1775 | 0.2466 | 0.2381 |
| 1892 | 0.1539 | 0.1259 | 0.1257 |
| 1893 | 0.4544 | 0.2897 | 0.3196 |
| 1894 | 0.3040 | 0.2739 | 0.2447 |
| 1895 | 0.6544 | 0.4181 | 0.3705 |

Entries are offline terminal position distances, lower is better. Spectral
is closer than zero on five roots and farther on three. Strict offline
distance `<0.5` is 6/8 in both arms: zero fails 1889 and 1895, while
spectral fails 1889 and 1890. Random also achieves 6/8. Root 1894, the
single root whose spectral prefix increased distance from the initial
position, has distances 0.2447, 0.2739 and 0.3040 for spectral, random and
zero, respectively. Its spectral cp5 is already closer than zero cp5, so
these numbers do not isolate a reconfiguration mechanism.

All three tails use the same optimizer, score, feedback opportunity and
48-sample budget, but prefix search and structure computation differ;
there is no certified equal-total-resource method comparison. The 16
random/spectral outcomes are prior results, not new executions. These roots
and methods were exposed during development; zero action can still lead to
movement due to velocity. No independent confirmation, greedy or strong
baseline comparison, non-greedy mechanism, or P1 gate is established.

- Report:
  `le-wm-main/results/a1_pointmaze_zero_prefix_proprio_replan_20261103_DEV01.json`,
  SHA-256 `13d69c90408483271f7db0d426a9411ab42035c71f60955a21a47a04eda35fef`.
- Arrays: same basename `.npz`, SHA-256
  `5e4ccc334598a5031d238af860189aaba7a412bedec23e2953319612948f4bb5`.
- Saved tail CEM, action, cp5 and outcome checks plus common-suffix checks:
  **2 passed**; tests do not rerun model inference or physics.
