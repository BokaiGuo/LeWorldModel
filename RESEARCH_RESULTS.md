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

## Permission-gated architecture result

The follow-up panel applies a frozen calibration-only permission gate using
horizon-normalized task RMSE and action-inverse risk. A rejected representation
is not counted as learned planning; it is forced to the fixed h=1 fallback.

| Frozen base | Seeds | Conservative gain | 95% interval | Sign-flip p | Permission |
|---|---:|---:|---:|---:|---|
| JEPA | 5 | 0.09803 | [0.06492, 0.13277] | 0.0306 | 25/25 accepted |
| DINO | 5 | 0.00000 | [0, 0] | 1.0000 | 25/25 rejected |

This is the current strongest interpretation: the module has a reproducible
local benefit on the JEPA representation and a safe, explicit rejection rule
for the DINO representation. It is not a universal architecture-independent
planning result.

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
