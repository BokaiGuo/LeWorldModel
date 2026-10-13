# PointMaze learned-adapter panel summary

Positive gain means the prediction-only diagnostic has lower held-out consequence loss than fixed h=1.

| Base | Adapter | Seeds | Raw gain mean [95% CI] | Conservative gain mean [95% CI] | Raw gate | Conservative gate | Permission |
|---|---|---:|---:|---:|---|---|---|
| dino | task_effect_mlp | 5 | -0.0417032 [-0.0824402, 0.00282798] | 0 [0, 0] | False | False | rejected_fail_closed:25 |
| jepa | task_effect_mlp | 5 | 0.174056 [0.162004, 0.186107] | 0.0980304 [0.0649238, 0.132769] | True | True | accepted:25 |

## Interpretation

The uncertainty unit is adapter seed; optimization stages are snapshots, not independent replications.
The conservative column is a calibration-only fail-closed gate; it may fall back to h=1 and is not pooled with the raw diagnostic result.
Planning permission is decided from calibration-only normalized task RMSE and action-inverse risk. A rejected candidate is not evidence of learned planning; it is an explicit safe fallback.
A negative DINO family result remains a representation-conditioned failure and is not pooled away by the positive JEPA family.

## Claim boundary

This is a PointMaze architecture-conditioned panel summary. It can diagnose a reproducible representation condition, but it does not establish cross-environment or universal transfer.
