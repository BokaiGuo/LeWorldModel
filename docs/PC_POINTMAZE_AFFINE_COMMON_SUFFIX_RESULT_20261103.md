# Affine-score prefixes: common-suffix physical intervention

The [fixed development protocol](PC_POINTMAZE_AFFINE_COMMON_SUFFIX_PROTOCOL_20261103.md)
executed 168 root/prefix/suffix combinations through the audited PointMaze
position backend: eight exposed roots, three five-step prefixes, and seven
shared suffixes evaluated at both five and ten subsequent high-level steps.
The prefixes are the previously selected affine-score spectral-CEM and
random-CEM prefixes and an all-zero prefix; no new model selection or
training was performed. Each saved own-prefix/own-tail five-step chain
reproduced its original endpoint position exactly.

| Root | Initial goal distance | Spectral cp5 | Zero cp5 | Random cp5 | Spectral cp5 − initial |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1888 | 0.4360 | 0.0650 | 0.0662 | 0.0699 | −0.3710 |
| 1889 | 1.1842 | 0.3286 | 0.3370 | 0.3797 | −0.8556 |
| 1890 | 1.1766 | 0.6486 | 0.6782 | 0.6245 | −0.5280 |
| 1891 | 0.7977 | 0.2515 | 0.2196 | 0.2296 | −0.5462 |
| 1892 | 0.3298 | 0.1391 | 0.1303 | 0.1395 | −0.1907 |
| 1893 | 0.6426 | 0.2392 | 0.4932 | 0.3677 | −0.4034 |
| 1894 | 0.2529 | 0.6233 | 0.6316 | 0.6628 | +0.3704 |
| 1895 | 2.2106 | 1.0446 | 1.3354 | 0.9904 | −1.1661 |

Only root 1894 meets this protocol's position-only non-immediate-progress
diagnostic for the spectral prefix. Even there, spectral cp5 is closer to
the goal than zero cp5; prefix position and velocity differ, so later
outcomes do not isolate a change in controllability.

For that root, the table shows the **same** suffix following spectral and
zero prefixes. The last two columns subtract each prefix's own zero-suffix
goal error at the corresponding horizon; lower is better:

| Suffix | Spectral final, 5 steps | Zero-prefix final, 5 steps | Spectral − own passive, 5 | Zero − own passive, 5 | Spectral − own passive, 10 | Zero − own passive, 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| +x | 0.2841 | 0.2680 | −0.0361 | −0.1191 | +0.3093 | +0.1753 |
| −x | 0.4616 | 0.4717 | +0.1414 | +0.0846 | +0.3080 | +0.1743 |
| +y | 0.2627 | 0.3187 | −0.0575 | −0.0685 | +1.3123 | +1.1694 |
| −y | 0.6338 | 0.6634 | +0.3136 | +0.2763 | +0.5099 | +0.4113 |
| Random arm's tail | 0.2406 | 0.2090 | −0.0796 | −0.1781 | +1.5198 | +1.3653 |
| Spectral arm's tail | 0.0369 | 0.0402 | −0.2833 | −0.3470 | +0.9666 | +0.8032 |

For all six nonzero suffixes, at both horizons the spectral prefix's
change relative to its own passive continuation is less favorable than
the zero prefix's corresponding change. Absolute endpoints sometimes
favor spectral, including the saved spectral tail at five steps, but the
prefix and passive continuation already differ. This is a negative result
for this fixed candidate/suffix diagnostic, not a universal impossibility
claim or a substitute for separately optimized suffix planning.

The first attempt was stopped by the historical-chain check because the
raw archive is float64 while the original dataset loader casts to float32.
After matching that loader conversion, all sixteen original own-arm chains
reproduced exactly. The failed attempt produced no result file and is an
interface/precision defect, not a method outcome.

Existing proprio-score common-suffix panels use different model-selected
actions on some of the same roots. Their overlap and this new panel are not
independent replications. The backend's contract certifies position labels;
saved velocities are diagnostic and must not be promoted to a general
cross-backend velocity certification. Neither this retrospective panel nor
the previous independently optimized tails establishes an independent,
equal-resource Planning Compiler advantage or opens P1.

- [Full per-root/prefix/suffix result](../le-wm-main/results/pc_pointmaze_affine_common_suffix_20261103.json),
  SHA-256 `5a0c331d2d88a0497c4e2f01c09d7f06b3841a832ec34fa6890241646e74d83e`.
- [Saved actions and physical states](../le-wm-main/results/pc_pointmaze_affine_common_suffix_20261103.npz).
- Affine, physics-matrix, score-replay and related tests: **53 passed**.
