# PointMaze C3 active-probe forensics: safety repair does not imply decision repair

## Status

This record separates one invalid opened panel (V5) from one fresh, strictly
valid confirmation (V6).  It is a protocol and mechanism record, not evidence
that an active local controller improves planning.

## V5 is invalid for efficacy claims

V5 completed calibration and all five held-out arm files but failed during
final validation.  Its raw trace contains an executed candidate-anchored
information action at held-out root `10035` with:

| field | value |
| --- | ---: |
| action-goal projection at execution | -0.14860 |
| predicted goal progress | +0.19569 |
| observed goal progress | -0.05372 |

Thus an initially screened pulse became goal-opposed after earlier probes and
was nevertheless executed.  The new read-only forensic audit hashes the
immutable V5 inputs and arm reports, runs the current strict validator, and
marks V5 as `invalid_opened_outcome_protocol_contract`.

Consequences are fixed:

- V5 cannot support a positive or negative efficacy claim.
- Its held-out outcomes cannot select a replacement threshold, gate, width,
  candidate schedule, or planner.
- V5 will not be rerun in place; any successor needs fresh roots and a new
  output path.

## V6 verifies the runtime safety repair, but not local decision transfer

V6 used fresh state roots, calibration roots disjoint from V5, and an
execution-time recheck before every physical pulse.  Its raw-trace audit
passes: all 135 executed candidate-anchored probe pulses have positive current
goal projection and positive predicted progress; two subsequently unsafe
pulses were rejected before execution.

The same panel nevertheless has no valid local decision release:

| quantity | V6 value |
| --- | ---: |
| held-out roots | 64 |
| roots with a triggered probe | 55 |
| candidate-anchored probe pulses | 135 |
| local verified releases | 0 |
| local rerank overrides | 0 |
| active minus HTSC final-distance difference | +0.05384 |
| active minus HTSC success difference | -0.12500 |

The paired 95% interval for active-minus-HTSC final distance is
`[+0.01143, +0.09958]`; for success it is `[-0.21875, -0.04688]`.
Accordingly, the active controller is not deployable.

## Mechanism localization

V6 makes a stronger diagnosis than a generic negative result.  Its
identifiability audit attributes the 64 roots to:

| post-probe failure mode | roots |
| --- | ---: |
| affine chart not identifiable | 28 |
| affine chart cross-fit unreliable | 15 |
| no selected candidate in local probe hull | 12 |
| probe not triggered or rejected early | 9 |

The mean final-candidate hull coverage is only `0.06989`.  In other words,
the safety repair correctly prevents stale backward probes, but a chart fit on
safe information actions almost never supports the later planning candidate
that HTSC has withheld.  Tightening or loosening the leave-one-out threshold
cannot solve that support mismatch without reopening a calibration question.

The completed balanced residual-bank calibration panel reaches the same
deployment boundary independently: every nonempty local gate is inferior to
the RGB visual baseline on its calibration roots, so its outcome roots remain
sealed.

## Consequence for the next protocol

The next *new* method must change information geometry, not merely the gate:

1. declare a candidate subset whose eventual local decision lies in the
   observed probe support (or use an explicitly validated interpolation rule);
2. reserve equal physical pulse budget for both probe and fallback arms;
3. choose that support construction and any threshold on calibration roots
   only;
4. open fresh roots only if the calibrated controller has a nonempty,
   noninferior release set; and
5. report chart acquisition, support coverage, local override count, ranking
   regret, and task outcome separately.

This condition avoids treating a safety-compliant no-override panel as proof
that verified local transfer is impossible, while also avoiding outcome-driven
retrying of V5 or V6.

## Reproducibility

- V5 forensic tool: `le-wm-main/experiments/audit_pointmaze_htsc_candidate_anchored_affine_probe_v5.py`
- V5 forensic artifact: `le-wm-main/results/pointmaze_htsc_candidate_anchored_affine_probe_v5_20261022_cuda/v5_forensic_audit.json`
- V6 report: `le-wm-main/results/pointmaze_htsc_candidate_anchored_affine_probe_v6_runtime_safety_20261022_cuda/report.json`
- V6 raw-trace audit: `le-wm-main/results/pointmaze_htsc_candidate_anchored_affine_probe_v6_runtime_safety_20261022_cuda/trace_contract_audit.json`
- V6 identifiability audit: `le-wm-main/results/pointmaze_htsc_candidate_anchored_affine_probe_v6_runtime_safety_20261022_cuda/identifiability_audit.json`
