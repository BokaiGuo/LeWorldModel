# PointMaze V20: budgeted prospective repairability

## Question

V19 showed that visible history motion predicts a large difference in the
value of recurrent repair, but its low-motion action-only fallback did not
beat an unconstrained always-recurrent policy.  V20 tests the decision that
actually remains unresolved: with a fixed, scarce repair budget, does the
predeclared visual diagnosis allocate recurrent repair to the roots where it
has more downstream value?

## Pre-outcome protocol

V20 creates a fresh pool after every V19 root and binds all root/action
identities against V19's sealed context.  Its 832 generated roots again
contain 448 position-matched opposed-velocity roots and 384 zero-velocity
controls.  The split is 256 train, 256 calibration, and 256 outcome roots,
with 128 roots per mechanism in every split.

The deployed policy and its equal-budget control are frozen before fitting:

| policy | recurrent repair condition | recurrent budget |
| --- | --- | ---: |
| proposed | visible XY motion `>= 0.02` | 50% |
| inverse control | visible XY motion `< 0.02` | 50% |
| action-only | never | 0% |

Both 50% policies use the same frozen recurrent/action models, candidate bank,
model width, history source, and task cost.  The inverse route is therefore a
control for allocating the repair budget, not a new learned selector.

## Admission and outcome criteria

The static ledger must establish fresh root and action identities, visible
causal histories, velocity reversal, rejection of a position-only proxy,
shared candidate support, and the exact inverse control.  Calibration can
authorize or reject only the already declared high-motion route.

The independent outcome audit declares a narrow C2 result only if both are
true on root-clustered held outcomes:

1. the fixed route improves regret over action-only; and
2. it improves regret over the equal-budget inverse route.

The audit also reports recurrent-versus-action benefit in each mechanism,
route-versus-always-recurrent regret, and route fractions.  It does not claim
C3 calibration, online MPC, contact transfer, or universal repairability.

## Sealed outcome

The host-CUDA run completed with 256 independent outcome roots after the
selection receipt had been frozen and hash-bound outcome labels were
generated.  Both route and inverse control invoked recurrent repair for
exactly 50% of roots.

| comparison | regret improvement | 95% paired bootstrap CI |
| --- | ---: | ---: |
| high-motion route vs. action-only | 0.09684 | [0.07734, 0.11606] |
| high-motion route vs. equal-budget inverse route | 0.09569 | [0.07656, 0.11527] |
| recurrent repair gain: high-motion roots | 0.19368 | [0.16241, 0.22462] |
| recurrent repair gain: direct-control roots | 0.00230 | [0.00123, 0.00351] |
| high-minus-direct repair-gain difference | 0.19138 | [0.16005, 0.22212] |

Thus V20 validates the intended narrow C2 claim: the visual-motion diagnosis
allocates a fixed repair budget to roots where repair has reliably more value
than under an equally costly inverse allocation.  It still does not beat
unconstrained always-recurrent repair (route-minus-recurrent regret gain
`-0.00115`, CI `[-0.00178, -0.00059]`), so it is not evidence that abstaining
from repair is intrinsically optimal without a budget constraint.

## Artifacts

- `results/pointmaze_mixed_velocity_repairability_v20_20261023_cuda/temporal_ood_v6_frozen_selection.json`
- `results/pointmaze_mixed_velocity_repairability_v20_20261023_cuda/v20_cuda_runtime.json`
- `results/pointmaze_mixed_velocity_repairability_v20_20261023_cuda/v20_outcome_root_cluster_audit.json`
