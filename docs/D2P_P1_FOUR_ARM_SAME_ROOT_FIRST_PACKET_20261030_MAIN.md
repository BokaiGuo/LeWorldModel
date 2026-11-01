# Four-arm first-packet prediction and physical comparison

This is a post-hoc comparison of four already executed actions on one known
root, not independent confirmation, an equal-budget comparison, or P1
admission. It joins the 16-node same-action CUDA result with the earlier
four-arm common-RGB physical report only at boundary 0 / step 5. The four
saved step-0 state vectors and task goals match. Each prediction uses its own
arm's actual five-step executed packet; the next RGB encoding and physical
state are from that arm's step-5 checkpoint.

| arm | predicted goal MSE | actual RGB goal MSE | object position error | wrapped angle error (rad) | joint position error |
| --- | ---: | ---: | ---: | ---: | ---: |
| structured | 1.043704 | 1.094152 | 70.7867 | 0.4759 | 160.0621 |
| native | 0.898127 | 0.727060 | 36.7523 | 0.2482 | 141.6599 |
| random | 0.909484 | 0.938622 | 70.7867 | 0.4759 | 136.5774 |
| immediate progress | 0.764582 | 0.606505 | 33.6747 | 0.2025 | 114.7711 |

All six pairwise predicted-versus-actual RGB goal-MSE orderings agree in this
first packet, despite nonzero prediction error. For object position and angle,
structured and random tie at the recorded physical precision while their
latent scores differ; the object does not move for either arm. This is a
concrete example of the latent metric distinguishing changes in the pusher
or image even when object pose is unchanged. The original native success
metric also involves the pusher, so object-only comparisons do not replace it.

This result does not rule out ranking errors at later boundaries: those
boundaries have different actual histories and states across arms. It also
does not establish the causal effect of the structured prefix, since the
four actions differ and future suffixes are not held fixed here.

Inputs:

- `codex_four_arm_same_action_prediction_20261030_v1_HOST01/result.json`,
  SHA-256 `ec7fd4edff5c673b8f4e42ddc66e88641d1616670fcda1b15efa2211a8f3ff51`.
- `codex_four_arm_common_rgb_launcher_20261030_RUN01/report.json`,
  SHA-256 `a035a1d6558ad88deebc8fbb6ce4384d012d7127a7cc615419b56948664ef2a3`.
