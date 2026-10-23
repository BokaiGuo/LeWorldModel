# CDR-WM Decision-Aliasing protocol

Status: `implemented audit; no fresh-panel result yet`.

This protocol implements B2 of the Counterfactual Decision Repairability
program.  It asks whether visually encoded histories that are close under a
fixed interface nevertheless induce materially different *realized candidate
task-contrast vectors*.

## Roles and timing

```text
calibration roots: representation-only radius calibration + realized contrasts
outcome roots:     post-execution realized contrasts only
```

The radius is computed from calibration representations without contrast
labels.  Candidate costs are centered inside each root, so the contrast vector
retains the complete candidate-preference geometry up to a shared additive
constant.  Calibration and outcome must use the same declared candidate-bank
hash and candidate identity sequence.

The PointMaze runner serializes outcome labels only after the repair router,
its threshold, and its independent certification decision are frozen.  The
alias audit runs after that execution point and cannot influence routing.

## Interpretation boundary

If a root-disjoint outcome history lies inside the frozen representation radius
of a calibration history while their realized contrast vectors differ by at
least the declared threshold, the result is **empirical near decision aliasing**.
It is evidence of a local mechanism that a history/belief repair should target.
It is not a causal-identifiability theorem or a conditional risk certificate.

Only exact representation collisions receive the elementary half-gap lower
bound for deterministic contrast readouts of that fixed interface.  Near
collisions receive no lower-bound claim without independently justified
regularity assumptions.

## V5 artifact contract

[`run_pointmaze_certified_repairability_v5_cuda.sh`](../le-wm-main/experiments/gdd_wm/run_pointmaze_certified_repairability_v5_cuda.sh)
performs, in order:

1. CUDA preflight and fresh V5 panel construction;
2. cross-fitted repairability routing and split certification;
3. post-freeze outcome execution and identity-bound label emission;
4. calibration-to-outcome aliasing audit.

The expected report is
`results/pointmaze_planwam_certified_repairability_fresh_v5_p2_20261021_cuda/decision_aliasing_calibration_outcome.json`.
Until it exists, the protocol supplies code coverage but no scientific result.
