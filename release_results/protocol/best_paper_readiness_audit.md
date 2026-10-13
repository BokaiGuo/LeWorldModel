# Best-Paper readiness audit

**Overall status:** `not_ready_without_independent_real_gain_and_cross_environment_calibration`

| Requirement | Status |
|---|---|
| reproducible transfer law | pass_synthetic_boundary_not_learned_generalization |
| reusable G_h/risk tool | validated_schema_and_fail_closed |
| independent real gain | blocked_missing_independent_gain_schema |

## Tool audit

Records: 90; fail-closed rejections: 90

The tool ranks checkpoints by `||G_h||` as a diagnostic proxy and rejects learned action release when independent gates are absent.

## Counterexample

Same declared spectrum, covariance, commutator, prediction MSE, and action inverse risk still yield different finite-horizon `G_h`.

## Independent gain gate

Status: `blocked_missing_independent_gain_schema`; valid explicit artifacts: 0

This artifact establishes a reusable G_h/risk-gate implementation and a finite-dimensional counterexample. It does not establish that G_h predicts learned planning risk until independent calibration and an external non-privileged held-out gain are supplied.
