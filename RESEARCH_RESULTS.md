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

## 2026-10-14 latest diagnostics

The final quality-matched PointMaze/PushT audit identifies 7 matched strata
over 2 environments, 4 architectures, and 5 seeds. The full diagnostic has
mean `Delta R=+0.05336` with paired bootstrap 95% CI `[+0.01150,+0.11661]`,
but the predeclared stratum sign-flip test gives `p=0.0630`; the positive
cross-environment gate therefore remains closed.

The new PushT contact-stratified intervention shows that contact geometry is
useful primarily in free/approach states; normal- and tangential-contact
within-mode action effects remain difficult to identify. A corrected
polygon-boundary signed-torque audit reduces privileged classifier angular
regret relative to the geometric baseline (`0.0151` vs `0.0265`), but reports
`0/32` strict pose successes and an angle-improvement interval crossing zero.
This is a mechanism diagnosis, not a pose-closure or visual-planning result.

The RGB PointMaze successor bridge is positive (`0.1631` RMSE versus `0.2011`
source-latent baseline), while direct history concatenation is not; causal
history remains useful only as a calibrated task-effect feature in the current
panel. See [docs/RESEARCH_PROGRESS_20261014.md](docs/RESEARCH_PROGRESS_20261014.md)
for artifact paths and full claim boundaries.
