# Curated experimental results

## PointMaze independent consequence

The corrected, pre-specified audit compares a fixed horizon-1 baseline with a
prediction-only diagnostic controller over a fixed candidate bank. The task
readout and latent center are fitted on train rows only; outcome rows are used
only for final scoring.

| Quantity | Value |
|---|---:|
| Paired outcomes | 32 |
| Mean baseline-minus-diagnostic loss | 0.17135 |
| 95% interval | [0.02309, 0.33601] |
| One-sided p-value | 0.02020 |
| Environments | 1 (PointMaze) |

This is a positive local result, not cross-environment evidence.

## Architecture boundary

The matched panel keeps the protocol fixed across JEPA/DINO frozen exports,
linear/MLP adapters, seeds, and training stages. The corrected panel shows
that JEPA adapters can yield positive local gains while DINO adapters remain
negative or inconclusive. The public conclusion is therefore representation-
conditioned action-to-task sufficiency, not a universal adapter guarantee.

## Theory and diagnostics

The public theory artifacts provide:

- a finite-horizon transfer-risk decomposition;
- local nonlinear applicability conditions;
- explicit spectrum/alignment/commutator counterexamples;
- a covariant metric-tube and fail-closed control audit.

The commutator is not presented as a universal transfer certificate. The
controller is permitted only when independently checked action
identifiability, transition consistency, and local risk-envelope conditions
are satisfied.
