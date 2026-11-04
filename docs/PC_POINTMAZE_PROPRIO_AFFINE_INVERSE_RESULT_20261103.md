# PointMaze fixed-bank affine-inverse diagnostic

The [development protocol](PC_POINTMAZE_PROPRIO_AFFINE_INVERSE_PROTOCOL_20261103.md)
evaluates an analytic, untrained projection of the frozen checkpoint's
predicted proprio features. The score uses the spatial mean of the predicted
16-dimensional proprio tokens, subtracts the encoded goal, applies the
Moore–Penrose inverse of the frozen 16-by-4 input encoder, rescales the first
two coordinates with the existing normalizer, and measures their norm.
Scores were saved before physical labels were read.

The encoder has full column rank, singular values approximately
`[0.47260, 0.31004, 0.05824, 0.03310]`, and condition number 14.28. The
maximum discrepancy between the saved goal token and the reconstructed
RGB-derived goal token was `5.26e-8`. All saved scores were finite. This
checks the binding for these inputs, not the provenance of the normalizer
or the accuracy of predicted physical states.

## Fixed candidate ranking

Eight groups contain 48 candidates each, on four previously exposed roots.
For each velocity input below, the table counts within-group score-order
reversals against physical endpoint position distance. The 9,024 pairs share
roots and trajectories and are **not independent samples**.

| Velocity input | Affine inverse | Visual latent | Proprio latent |
| --- | ---: | ---: | ---: |
| Zero | 2,940 / 9,024 | 3,383 / 9,024 | 2,994 / 9,024 |
| Causal RGB estimate | 641 / 9,024 | 2,229 / 9,024 | 2,894 / 9,024 |
| Real velocity, diagnostic only | 696 / 9,024 | 2,318 / 9,024 | 3,045 / 9,024 |

Against actual endpoint RGB latent cost instead, causal affine, visual and
proprio scores respectively have 2,412, 1,026 and 3,677 reversals. These
are different reference objectives; disagreement does not itself establish
a coding error.

The sealed ranking implementation determines reversals by multiplying two
score differences. An independent test found that this can underflow for
extremely small differences. A separate sign-comparison implementation
recomputed all three score variants, all velocity modes, both references,
and all eight groups: its counts agree with the sealed result on this
particular dataset. The historical analysis is unchanged; future ranking
analyses should use the numerically safe implementation.

For the causal velocity input, the physical distance of the selected
candidate is:

| Root and candidate bank | Affine inverse | Visual | Proprio |
| --- | ---: | ---: | ---: |
| 1884 random CEM | 0.0241 | 0.1454 | 0.7374 |
| 1884 spectral CEM | 0.0523 | 0.0419 | 0.4003 |
| 1885 random CEM | 0.1991 | 0.1991 | 0.3291 |
| 1885 spectral CEM | 0.1127 | 0.1127 | 0.3560 |
| 1886 random CEM | 0.2099 | 0.2193 | 0.2605 |
| 1886 spectral CEM | 0.0463 | 0.2193 | 0.2026 |
| 1887 random CEM | 0.2124 | 0.3257 | 0.2124 |
| 1887 spectral CEM | 0.1717 | 0.1717 | 0.4938 |

Affirmed per group, affine versus visual is 4 lower, 3 equal and 1
higher; versus proprio it is 7 lower and 1 equal. These are descriptive
comparisons on shared, exposed roots, not independent inference.

For causal velocity, group mean projection residual norms range from
0.00469 to 0.00974; group mean patch-spread RMS ranges from 0.00256 to
0.00399. The residual is calculated for the mean prediction **difference**
from the goal, not the absolute affine-space error. Neither quantity has a
calibrated physical error scale, and neither certifies an accurate decoder.

## Interpretation and limits

The four-to-sixteen input map is affine; model-predicted patch tokens are
not guaranteed to lie in its affine image or agree across patches. This
projection is a testable scoring construction, not a trained or validated
state decoder. Each candidate here is evaluated from its **own actual cp5
image**, which is an oracle replay unavailable when choosing among future
prefixes. Real velocity is privileged and is excluded from deployment.
The RGB goal-to-coordinate conversion relies on camera/scene calibration,
which must be declared and audited before a deployment claim. The normalizer
comes from the whole local archive; its relationship to checkpoint training
data remains unknown.

This offline result motivates a separately specified real-feedback
experiment: after one actually selected prefix, compare score variants
using that arm's own causal RGB history, matched optimization and resource
limits. It does not establish closed-loop advantage, a reconfiguration
mechanism, spectral advantage, independent generalization, or PC/P1 admission.

- [Score receipt](../le-wm-main/results/pc_pointmaze_proprio_affine_inverse_20261103_scores.json)
- [Scored arrays](../le-wm-main/results/pc_pointmaze_proprio_affine_inverse_20261103_scores.npz)
- [Physical and RGB evaluation](../le-wm-main/results/pc_pointmaze_proprio_affine_inverse_20261103_evaluation.json)
- No new neural inference or physical execution was performed for this diagnostic.
