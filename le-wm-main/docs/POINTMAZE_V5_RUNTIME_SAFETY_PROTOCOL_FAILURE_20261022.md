# V5 candidate-anchored probe: runtime-safety protocol failure

## Status

`pointmaze_htsc_candidate_anchored_affine_probe_v5_20261022_cuda` is an
**invalid efficacy panel**.  It must not be used as evidence that observed
local action-effect charts help or harm planning.

All five raw held-out arm reports were written, but the strict aggregate
correctly aborted before `report.json` because a candidate-anchored information
action violated the declared task-safety contract.

## Failure mechanism

V5 selected its complete candidate-anchored probe schedule at the episode
start.  Earlier probes changed the live RGB-decoded position, but the later
scheduled pulses were executed without rechecking their task alignment.

For held-out root `10035`, the fourth observed probe was candidate index `1`,
with action `[0.19567585, -0.15559871]`.  At execution it had:

| quantity | value |
| --- | ---: |
| action projection toward current target | `-0.14860218` |
| predicted goal progress | `+0.19568889` |
| observed goal progress | `-0.05371817` |

The negative live action projection violates the V5 requirement that every
executed information action be goal aligned.  The mismatch arises because the
predicted progress and candidate selection were calculated at the initial
state, while the logged projection was evaluated after three state-changing
probes.

## Remediation and fresh confirmation

The runtime now recomputes both the current-target action projection and the
calibration-dynamics one-block progress immediately before every probe.  If
either is nonpositive, the pulse is not executed, the probe sequence stops,
and the unspent physical action budget returns to ordinary planning.

The repair is tested in `tests/test_live_pointmaze_planning.py`.  V6 runs the
corrected protocol on a new independently generated state/action extension
(`12920–13127`), V5-disjoint calibration roots (`1832–1863`), and 64 fresh
held-out roots (`12920–12983`).  Its evidence is valid only if the raw-trace
contract audit passes after the calibration-frozen configuration is evaluated.

## Claim boundary

This failure identifies an implementation/protocol defect, not a negative
scientific result about local observed-effect adaptation.  V6 is a new
experiment; it is not a re-analysis or a relabeling of V5 outcomes.
