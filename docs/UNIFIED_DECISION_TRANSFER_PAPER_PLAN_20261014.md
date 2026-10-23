# Unified decision-transfer paper plan

## Research spine

This workstream integrates ATM, HTSC, and GDD-WM as one falsifiable chain:

1. **Diagnose**: action identifiability does not imply decision transfer.  The
   primary object is the finite-horizon task contrast
   \[
   \Delta^*_{ij,H}=J^*_H(h_t,U_j,g)-J^*_H(h_t,U_i,g),
   \]
   together with its model estimate and contrast error.
2. **Repair**: only after an execution-grounded teacher preference is available,
   test identifiability-guided predictive-geometry distillation (GDD-WM).
3. **Decide**: after the model and readout are frozen, calibrate a
   horizon-conditioned selective certificate (HTSC).  A candidate is released
   only when its predicted margin exceeds the independent calibration envelope
   and the action/state support gate passes; otherwise the controller probes or
   abstains.

## Non-negotiable separation

Teacher preference data, calibration roots, and held-out closed-loop outcomes
are mutually exclusive.  Any encoder, predictor, metric, or GDD-WM student
change invalidates the old HTSC calibration and requires recalibration.  The
PointMaze candidate bank is currently an offline counterfactual benchmark: it
does not contain fresh RGB realized successors for every candidate and therefore
cannot authorize visual online planning.

## Evidence gates

| Gate | Object | Passing evidence | Current status |
|---|---|---|---|
| P0 | candidate/task contrast ceiling | independent root-disjoint contrasts are measurable and provenance-complete | partial: measurable, ranking is imperfect |
| P1 | ATM/MSE decision-transfer baseline | matched diagnostic has independent planning consequence | partial/gated: current matched gate does not pass |
| P2 | execution-grounded GDD-WM | frozen student improves held-out task transfer, not only teacher agreement | not run / proposed |
| P3 | HTSC calibration | support-aware risk-coverage and regret on held-out roots | fail-closed: support unavailable in current bank |
| P4 | RGB-only online MPC | independent visual closed-loop consequence at declared coverage | blocked |

## Current scientific estimands

The confirmatory estimand is

\[
\Delta R\mid \mathrm{MSE},\mathrm{spectrum},\mathrm{stage},
\mathrm{action\ risk},
\]

with root/model-seed/environment as the inference unit.  A lower prediction
error without lower action-ranking regret is a boundary result, not a success.
Alignment, commutator, Schur geometry, resolvent, and transient gain remain
diagnostics; none is a universal transfer certificate alone.

## Current P0/P1 execution record

The reproducible CPU post-hoc rerun is written under
`le-wm-main/results/unified_decision_transfer_p0p1_20261014/`.  Its manifest
records the GPU preflight failure, input fingerprints, exact commands, and the
claim boundary.  The expected interpretation is deliberately conservative:
an offline task-contrast artifact can establish the decision-transfer gap and
the need for HTSC, but cannot establish GDD-WM, frozen RGB planning, or a
universal world model.

## Failure boundaries retained as results

- finite-horizon task ranking can fail even when a latent successor probe is
  informative;
- support-unavailable certificates must abstain rather than shrink the bound;
- mode or prediction improvements without within-mode action identifiability do
  not establish pose closure;
- GDD-WM remains `Proposed / Not Yet Validated` until its teacher provenance,
  student-generated candidates, and independent planning consequence are all
  present.

## PlanWAM-relative execution update — 2026-10-23

PlanWAM is treated as a strong planning-shaped future representation plus
candidate-scoring reference, not as a failed baseline. The next claim is
narrower: pre-outcome diagnostics should predict whether a learned visual
world-model decision error is repairable under a declared support and repair
function class.

The paper is organized as one regularity with a contact-domain boundary:

- PointMaze V17/V18 supplies the replicated renderer-visible velocity-alias
  mechanism in which history adds fixed-bank action-ranking information.
- PointMaze V7/S1-S3 are prospective router protocols. Their raw outcome
  diagnostics are not promoted to efficacy evidence unless the calibration
  certificate and independent outcome audit both pass.
- TwoRoom V4 is the independent contact/action-effect replication. Its matched
  controls are action-only and no-dynamics, and its three panels are the
  independent replication units. TwoRoom remains a boundary test, not a
  guaranteed second positive method result.
- PushT remains an orientation/contact stress test; offline mechanics and
  contact retention cannot substitute for strict pose closure.

The V4 aggregate now reports both root-paired uncertainty nested within each
panel and low-resolution panel-level inference. It also rejects any root reuse
across train, calibration, or outcome roles, including reuse across panels.
The fresh host-CUDA panels have now completed: the full adapter improved
terminal distance in all three panels, but worsened root-paired candidate
ranking regret versus both matched controls and did not improve success. With
only three independent panels, its panel-level sign-flip p value is 0.25.
Therefore TwoRoom is retained as a contact/action-effect boundary, and V5
support-gate tuning on these opened outcomes is forbidden.

PointMaze now has a separate prospective comparison-matrix audit contract:
the route and all candidate scores must be sealed before outcome generation,
then compared root-paired against uniform history repair, ordinary preference,
and visible-motion/MSE/support/ATM controls with Holm correction. This fills a
protocol gap but has no fresh-root result yet.
