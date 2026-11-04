# Fixed-action cp12 real-RGB refresh: development result

All 32 saved executed tails were evaluated with the same actions: open
prediction from cp7 for ten actions, and prediction from the actual cp12
RGB-derived state for the remaining five. Open scores reproduced the saved
feedback scores, and the actual cp17 scores reproduced the source report.
No new actions were executed.

| Measure | Open mean | Refreshed mean | Refresh lower |
| --- | ---: | ---: | ---: |
| Visual latent MSE | 0.0268260 | 0.0241279 | 21/32 |
| Proprio latent MSE | 0.00021620 | 0.00018300 | 16/32 |
| Absolute task-score error | 0.0667291 | 0.0403471 | 19/32 |

Mean reductions are descriptive over candidate trajectories sharing four
exposed roots, not independent statistical confirmation.

| Root | Open vs physical reversals | Refresh vs physical reversals | Open vs RGB-score reversals | Refresh vs RGB-score reversals |
| ---: | ---: | ---: | ---: | ---: |
| 8343 | 5/28 | 3/28 | 4/28 | 2/28 |
| 8345 | 5/28 | 6/28 | 4/28 | 5/28 |
| 8349 | 2/28 | 0/28 | 2/28 | 0/28 |
| 8351 | 4/28 | 2/28 | 3/28 | 1/28 |

Root 8345 has one tie in actual RGB score. Refresh lowers aggregate error
but worsens both rankings on that root, illustrating that improved
prediction error does not guarantee improved action ordering. The task score
uses RGB-estimated state through a proprio encoder and affine inverse;
physical position is evaluation-only.

This is fixed-action teacher-forced analysis: refreshed context comes from
real cp12 observation, but no action is replanned at cp12. It therefore does
not demonstrate improved closed-loop control, isolate model error from RGB
state estimation or timing, or establish a non-greedy mechanism. The
checkpoint's original training/normalizer provenance and the roots'
independence are not certified.

Counts: 97 encodes, 32 two-packet and 32 one-packet model samples, no new
physical steps. A saved-output audit recomputed all 32 embedding MSE values,
source-score bindings and rankings. It did not independently rerun inference.

- Protocol: `docs/PC_CAUSAL_MIXED_CP12_REFRESH_PROTOCOL_20261103.md`.
- Runner: `le-wm-main/experiments/pc_causal_mixed_cp12_refresh_20261103.py`.
- Results and saved features:
  `le-wm-main/results/pc_causal_mixed_cp12_refresh_20261103_DEV01.json`
  and the same-name NPZ.
- Audit: `le-wm-main/results/pc_causal_mixed_cp12_saved_audit_20261103_DEV01.json`.
