# Dev304 RGB-goal-tangent weighted spectral development experiment

This development experiment tests the existing RGB-only goal-tangent
metric in the same two known-root feedback contexts used by the unweighted
goal-contribution run. The frozen encoder builds the metric from the goal
image and fixed image perturbations; no simulator state or outcome enters
the model-side metric or selection. The same square-root metric weights the
Jacobian and residual for SVD mode ranking, and weights the nonlinear
terminal residual for candidate selection. Actions remain bounded and the
model's parameters are unchanged. This does not establish that the metric
captures the task or that the full compiler is RGB-only in every deployment.

All 18 predeclared candidates were physically executed to the ten-step
tail horizon; none succeeded natively. The saved verifier checks candidate
actions, prefix replay, 252 RGB frames, frozen model hashes, the weighted
mode order, and independently recomputes weighted candidate scores.

| Context | Candidate | Weighted model score | Native reward | Position error | Angle error |
| --- | --- | ---: | ---: | ---: | ---: |
| original | anchor | 1.8169 | -312.309 | 207.536 | 2.7110 |
| original | selected spectral | 1.0733 | -315.422 | 210.565 | 2.8070 |
| coherent | anchor | 3.6808 | -295.760 | 205.792 | 2.9933 |
| coherent | selected spectral | 2.6003 | -324.170 | 222.198 | 3.0058 |

In both contexts the selected action has a lower predicted weighted score
but worse physical position, angle, and reward than its own anchor. These
contexts share one development root, and scores are not directly comparable
across different metrics. This is neither proof that all task metrics fail
nor a unique attribution to prediction, scoring, contact or interface.
The end-to-end upstream resource cost is not certified; there is no P1/PC2
admission, independent confirmation, or closed-loop advantage here.

Artifacts relative to `le-wm-main/results/`:

- Model contract SHA-256 `2200a9b5e419718c6dda1f6dc7dac8d8a540dd5fb6c64bf3156e756db9092f7e`.
- Model report `p1_dev304_qgoal_spectral_20261102_RUN_DEV01/report.json`,
  SHA-256 `a0b61abd8d88ba0dccd884e37f42fc13b8e1d34e16af8ba4a7a91b21ff826349`.
- Physical contract SHA-256 `e937e573bdfff4cbd68b08cb5e7d5e96916ebbae7cb1b79fa72aa0526955237c`.
- Physical report `p1_dev304_qgoal_spectral_physical_20261102_RUN_DEV01/report.json`,
  SHA-256 `f87394ee94d9323fcf83caf66f01c1606c52124ad490e392dddc8b5202d7e642`.
- Verification `p1_dev304_qgoal_spectral_saved_20261102_DEV01.json`,
  SHA-256 `d30aa99a2157ee11bc09af44187b305a39d73136b957e6b2fa1afef8cb8969d6`.
