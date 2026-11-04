# Failure banks: model prediction versus real RGB versus physical distance

The fixed protocol `PC_POINTMAZE_FAILURE_REAL_RGB_METRIC_PROTOCOL_20261103.md`
replayed and encoded the actual terminal RGB of all 624 sealed candidates.
Each physical terminal position exactly matched the prior saved oracle.
The same frozen model, goal image and initial mask were used; the model
parameter hash was unchanged. These are post-outcome candidate-bank
diagnostics on three exposed roots selected after failure.

| Root / arm / bank | Model vs real RGB reversed | Real RGB vs physical reversed | Model-choice distance | Real-RGB-oracle choice distance | Physical-oracle distance |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1889 random initial | 1012/2016 | 938/2016 | 1.247 | 0.139 | 0.139 |
| 1889 spectral initial | 485/1128 | 594/1128 | 1.262 | 0.147 | 0.139 |
| 1893 random initial | 959/2016 | 980/2016 | 1.281 | 0.065 | 0.065 |
| 1893 spectral initial | 503/1127 | 647/1127 | 1.236 | 0.237 | 0.237 |
| 1895 random initial | 891/2016 | 323/2016 | 0.469 | 0.060 | 0.060 |
| 1895 spectral initial | 592/1128 | 333/1128 | 0.611 | 0.064 | 0.064 |
| 1889 random tail | 348/1128 | 699/1128 | 1.169 | 1.321 | 0.867 |
| 1889 spectral tail | 435/1128 | 702/1128 | 1.165 | 1.305 | 0.863 |
| 1893 random tail | 112/1128 | 769/1128 | 1.320 | 1.249 | 0.470 |
| 1893 spectral tail | 200/1128 | 544/1128 | 1.176 | 1.176 | 0.783 |
| 1895 random tail | 252/1128 | 109/1128 | 0.208 | 0.112 | 0.099 |
| 1895 spectral tail | 287/1128 | 580/1128 | 1.218 | 1.132 | 1.106 |

On all six initial banks, scoring the *actual future image* selects a
candidate with physical distance below 0.5, while the original
model-selected complete plan does so in only one. This supports a
prediction-to-actual-RGB mismatch in these exposed initial banks, but the
actual future is unavailable when selecting an action.

The same intervention does not fix the fixed-prefix tail cases. For example,
on 1893 random tail the model/real-RGB ranking has 112/1128 reversals, yet
real-RGB/physical ranking has 769/1128; the true-RGB-cost argmin is at
distance 1.249 whereas an available tail is at 0.470. This is direct evidence
that the chosen latent goal score can disagree with the physical objective
even on real images. It is not proof that the world model is otherwise
correct: same-action comparisons are rank summaries and the RGB/physical
measurement and task semantics still require scrutiny.

Actual-RGB and physical argmins are **privileged future oracles**, not
deployable policies. Candidate pairs share trajectories, CEM banks are
adaptive and roots are outcome-selected; no independent statistics or P1
claim follows. Fixed-prefix tail and full open-loop candidate banks cannot
be substituted for one another.

Artifacts:

- JSON `le-wm-main/results/a0_pointmaze_failure_real_rgb_metric_20261103_DEV01.json`,
  SHA-256 `714caf46a32abc659934870c9edfac8b442e53f95718e5fa8b29663bf2c2afc3`.
- NPZ `le-wm-main/results/a0_pointmaze_failure_real_rgb_metric_20261103_DEV01.npz`,
  SHA-256 `fac82371363e3881c044c946850e28dfa103402957cf38740e2e6c7394df7f36`.
- Saved ranking, position and oracle bindings: **1 passed**. The test does
  not repeat neural inference or physical rendering.
