# Real-RGB pixel metric: six-chain development diagnostic

The fixed protocol evaluates actual final frames from the six previously
executed chains against each root's recorded goal frame. Full-image RGB MSE
and MAE are computed without training, masks or tuned weights. The protocol
was written after the physical and latent results had already been seen.

| Root | Metric | Chosen arm | Physical regret | Physical-order concordance |
| --- | --- | --- | ---: | ---: |
| 20263160 | Predicted latent | selected | 0.088589 | 1/3 pairs |
| 20263160 | Actual latent | selected | 0.088589 | 2/3 pairs |
| 20263160 | Actual RGB MSE / MAE | same-bank greedy | 0 | 3/3 pairs |
| 20263161 | Predicted latent | selected | 0.019814 | 2/3 pairs |
| 20263161 | Actual latent | selected | 0.019814 | 1/3 pairs |
| 20263161 | Actual RGB MSE / MAE | same-bank greedy | 0 | 2/3 pairs |

Regret is final joint distance minus the lowest distance among the three
fixed arms for that same root. No pair was tied. Both pixel metrics select
the physically best arm in these two exposed cases, but the second root
still contains a pixel-versus-physical ordering reversal. This small,
post-exposure analysis is not evidence of general metric superiority.

Critically, these are **actual post-execution images**. LeWM does not supply
these images before choosing the candidate, so this calculation is an
oracle-like evaluation diagnostic, not a deployable pixel-scored planner
and not proof that a new selector will improve control. It does motivate
testing an allowed RGB-derived task signal prospectively, with separate
development/confirmation roots and real feedback.

The input action/physical/RGB receipt hashes and source journal are checked,
and a test fully recomputes the saved report from the bound source frames.
Eight targeted tests passed. The first output attempt failed during JSON
serialization because NumPy integer counts were not native integers; it
performed no physical or neural work and its partial output is retained as
`reacher_actual_rgb_pixel_metric_20261102_v1.json`. The fixed result was
written separately as `reacher_actual_rgb_pixel_metric_20261102_v2.json`
with SHA-256
`a9260d91b67c94dc69427129fd9c54530611cd3b95c72146acd1d14d98e9855f`.
No independent root, equal-budget baseline, mechanism intervention, or P1
admission is established.
