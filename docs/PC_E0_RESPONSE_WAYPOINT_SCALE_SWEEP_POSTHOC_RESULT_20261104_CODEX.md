# E0 response scale sweep: post-hoc diagnostic

## Scope and result

We replayed 48 previously optimized, float32 five-step directions from eight
exposed roots (three waypoint arms, two branches). This is a fixed post-hoc
analysis, not a new search, an independent task sample, or PC1 certification.
The original nominal is repeated `[0.25, 0]`; the saved coordinate central
finite-difference response predicts endpoint displacement along each action
direction. Interpolated actions are rounded to float32 before prediction and
physical replay. Relative error is endpoint prediction error divided by actual
endpoint displacement, and is undefined when that displacement is near zero.

| Fraction of saved direction | Median max action displacement | Median absolute error | Median relative error | Correct goal-change sign | Changed collision branches |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.0382 | 0.0020 | 0.0516 | 46/48 | 0/48 |
| 0.10 | 0.0764 | 0.0079 | 0.1027 | 47/48 | 0/48 |
| 0.25 | 0.1910 | 0.0492 | 0.2586 | 47/48 | 8/48 |
| 0.50 | 0.3819 | 0.2283 | 0.5225 | 41/48 | 13/48 |
| 1.00 | 0.7638 | 0.7551 | 0.9177 | 34/48 | 3/48 |

These are direction-level descriptive summaries, not independent-sample
confidence intervals. At full scale, the range of the eight within-root
median relative errors is 0.358–1.751. Collision changes are not monotone
across scales; a full-scale endpoint can return to the nominal branch.
The limited full-scale branch changes do not imply a complete causal
decomposition of the nonlinear errors.

All 48 saved coordinate-probe sets and all 48 directional ±0.001 probes
followed the nominal blocking sequence. Maximum Frobenius discrepancy of the
saved coordinate finite-difference matrix from the fixed-branch analytic
Jacobian was 0.000664; maximum directional finite-difference discrepancy was
0.00000722. These are absolute errors and include finite-difference and
float32 effects. The analytic Jacobian is valid only on a locally stable branch.
Separate float64 finite-difference tests passed on one unblocked and one blocked
fixed-branch example. This is not an AD/JVP verification or a learned-model
physical-response validation.

## Audit and interpretation

The analysis compares the exact backend against an independently coded
line-segment collision replay. A second script reloaded source actions and
recomputed endpoints, linear predictions, relative errors, sign comparisons,
branch changes and principal aggregates; all checked values agreed. This is a
same-backend consistency audit, not an independent scientific replication.

The observations support a narrow conclusion: on these selected E0 directions,
small-step response and fixed-branch derivative agree, while extrapolation to
the selected large actions is often inaccurate. They do not establish a safe
trust-region cutoff, the cause of planner failure, or whether a shrink/relinearize
policy improves planning under equal resources. Any such policy needs a new
version, preregistered development protocol, full cost accounting and new roots.
The stopped source panel remains stopped.

## Artifacts

- Protocol: `docs/PC_E0_RESPONSE_WAYPOINT_SCALE_SWEEP_POSTHOC_PROTOCOL_20261104_CODEX.md`
- Raw results and dependency hashes:
  `le-wm-main/results/pc_e0_response_waypoint_direction_dev_20261104_codex/scale_sweep_posthoc.json`
- Same-backend arithmetic audit:
  `le-wm-main/results/pc_e0_response_waypoint_direction_dev_20261104_codex/scale_sweep_posthoc_audit.json`
- Fixed-branch tests:
  `le-wm-main/tests/test_pc_e0_scale_sweep_jacobian_20261104_codex.py`
