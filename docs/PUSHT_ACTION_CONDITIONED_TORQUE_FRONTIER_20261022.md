# PushT action-conditioned signed torque frontier

## Question

Can a runtime visual descriptor identify which of a matched positive- or
negative-moment contact pulse will yield the more positive, zero-drift-corrected
angular response?  The target is deliberately narrower than pose planning.  A
positive result would authorize only a subsequent two-action ranking bridge;
it would not establish closed-loop pose closure or a general world model.

All panels below use three independent high seed bands, root-disjoint
train/calibration/test cohorts, balanced offline labels before fitting, and
same-reset zero/plus/minus real successors used only to form the label.  CUDA
work was executed by a host `systemd --user` service through
`run_with_torch_cudnn.sh`; the Codex sandbox was not used as a CUDA runtime.

## What the repair changed

The first signed contact-frame protocol passed RGB history and prior actions
to the readout, but omitted the candidate pair whose effects were being
compared.  This is an action-conditioning contract error: the deployed object
is `p(effect | RGB history, candidate action)`, not `p(effect | RGB history)`.

`run_pusht_action_conditioned_signed_torque_confirmation.py` therefore adds
the legal `plus`/`minus` action vectors to an RGB-only geometry history:

1. mask moments of pusher, movable T, and target;
2. an RGB pusher-to-block/nearest-block-pixel interface chart;
3. declared action history and the current candidate pair;
4. RGB lever-arm × action cross/dot terms; and, in V3, directed T body-frame
   projections using second- and third-order mask moments.

No runtime simulator state, contact label, successor image, or goal-derived
preference is an input.

## Independent held-out panels

| panel | confirmation arm | held-out BA | 95% CI | paired gain vs geometry | status |
|---|---|---:|---:|---:|---|
| V1 | geometry + raw candidate vectors | 0.461 | [0.402, 0.520] | +0.004, [-0.027, +0.035] | not confirmed |
| V2 | geometry + RGB lever-arm/action mechanics | 0.648 | [0.590, 0.707] | +0.074, [+0.016, +0.133] | not confirmed |
| V3 | geometry + lever-arm + directed body-frame mechanics | 0.621 | [0.563, 0.680] | +0.031, [-0.023, +0.086] | not confirmed |

The predeclared confirmation rule requires all of:

```text
held-out BA >= 0.70
held-out 95% lower bound >= 0.60
paired gain lower bound > 0
```

V2 is an exploratory positive direction only.  Its numerical paired delta was
computed from the mechanics arm, but the emitted `direction` text inherited an
older raw-pair label.  The code contract was corrected before V3; V2 must not
be used as formal confirmation or to select a controller.

## Interpretation and gate

The mechanics chart is the first repair that produced a positive paired gain
on an independent held-out panel.  The V3 replication did not preserve a
strict paired gain or meet the absolute criterion.  Thus the supported result
is **mechanistic signal, not stable action identifiability**.

Do not train a PushT runtime selector or launch a learned pose-closure claim
from these panels.  The next admissible intervention must add information that
is not a further reparameterization of passive RGB: a predeclared short probe
followed by a new observation, or a separately validated multi-view/contact
sensor interface.  It must then pass a fresh root-disjoint signed-pair
authorization before any closed-loop comparison.

## Artifacts

- V1: `results/pusht_action_conditioned_signed_torque_confirmation_v1_20261022_cuda/action_conditioned_signed_torque_confirmation.json`
- V2: `results/pusht_action_conditioned_signed_torque_confirmation_v2_mechanics_20261022_cuda/action_conditioned_signed_torque_confirmation.json`
- V3: `results/pusht_action_conditioned_signed_torque_confirmation_v3_bodyframe_20261022_cuda/action_conditioned_signed_torque_confirmation.json`
