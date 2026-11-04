# Dev304: goal-mode ranking implementation discrepancy

The Planning Compiler proposal ranks spectral modes by their contribution to
the damped local task objective, `sigma²/(sigma²+lambda) * (pᵀr)²`. The
sealed dev304 feedback spectral experiment instead ranked absolute damped
action coefficients, `|sigma/(sigma²+lambda) * pᵀr|`. These quantities are
not equivalent: a large action coefficient need not yield the largest
decrease in the damped local objective.

We added a separate numerical implementation and recomputed both rankings
from the sealed model Jacobians and RGB goal encoding, without changing the
historical experiment. Both use identity latent and action metrics and
`lambda=0.01`. The sum of the two selected mode contributions is:

| Feedback context | Existing coefficient modes | Goal-contribution modes | Existing sum | Goal sum |
| --- | --- | --- | ---: | ---: |
| original | 14, 16 | 6, 3 | 13.2756 | 42.0863 |
| coherent | 15, 13 | 9, 5 | 14.6205 | 38.7077 |

These are **unconstrained local quadratic** objective units, not actual
clipped rollout improvements, actual task costs, or independent evidence.
The new modes have not yet been used to generate, predict, or physically
execute candidates. The discrepancy is a credible implementation issue to
test, not proof that it explains the original physical failure.

The new routine takes an already task-weighted Jacobian and residual in
already whitened action coordinates. It neither applies the metrics itself
nor certifies nonlinear behavior. Tests check the analytical quadratic
improvement, a case where the rankings differ, invalid inputs, zero modes,
and sealed panel bindings.

The exploratory saved analysis is
`le-wm-main/results/p1_dev304_mode_rank_saved_20261102_DEV01.json`, SHA-256
`c75881427ccd1caafec79d7cd516c620f0d1842b908be062e2834383bceb5187`.
Before using the corrected ranking as a planner change, freeze a new
development protocol and compare both rankings using the same bounded
candidate generation, nonlinear rollout, physical execution, and accounted
budget. Preserve the original result and do not treat this post-outcome
analysis as independent confirmation.
