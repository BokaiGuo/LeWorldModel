# PushT learned-adapter negative-result audit and repairs

This report records the 2026-10-10 matched experiments. Existing negative
results are retained; none of the new runs overwrites or replaces them.

## Fixed protocol

- RGB/history/goal/action are the only runtime inputs.
- Simulator state and successors are used only for offline intervention labels
  and evaluation.
- `data-seed=73001`, `action-repeat=2`, 1,600 training roots, 320 held-out
  roots, eight-direction counterfactual fan, and the same bootstrap checkpoint
  are used across the matched variants.
- The final closed-loop check uses fresh `eval-seed=20411`, 64 episodes, and
  no runtime successor verification.

## What caused the negative result

1. Mode-conditioned dynamics were trained with oracle one-hot contact modes but
   evaluated with predicted soft modes. The interface audit identified this
   train/runtime mismatch. Training with the runtime soft mixture improved
   action-motion ranking, but oracle routing did not improve orientation
   ranking after the repair; routing was therefore a contributing cause, not
   the remaining main bottleneck.
2. The ordinary six-dimensional state readout had poor RGB orientation
   recovery under contact occlusion. On the repaired seed-8411 checkpoint,
   current angle RMSE was 1.451 rad before a dedicated history readout.
3. The goal-image angle readout remained poor even after a separate goal-angle
   head: 1.606 rad on the seed-9173 held-out audit. Thus the terminal planner
   can receive a geometrically plausible position while still optimizing
   against an unreliable target orientation.
4. The learned utility head was not the same object as the explicit successor
   angle-effect head. The utility terminal-phase ranking underperformed the
   held-out successor orientation ranking, so a successor-oriented phase
   interface was added and evaluated separately.

## Repairs and outcomes

| Variant / intervention | Contact orientation top-1 | Contact motion top-1 | Strict RGB-only success |
| --- | ---: | ---: | ---: |
| Previous h=2 reference, seed 8411 | 0.227* | 0.600 | 0/64 |
| Soft mode + explicit angle effect, seed 8411 | 0.320 | 0.650 | not run |
| + signed torque feature, seed 8411 | 0.213 | 0.577 | not run |
| + history angle readout, seed 8411 | 0.437 | 0.697 | 1/64 direct; 5/64 task-space |
| + history angle readout, seed 9173 | 0.403 | 0.717 | 8/64 task-space |

`*` The reference value is the first comparable explicit-angle audit; earlier
artifacts did not report this metric.

The history readout is a genuine offline improvement and replicates across two
model seeds. It does not yet establish stable posture closure: the best matched
closed-loop result is 8/64, with mean terminal angle error 1.511 rad and mean
block XY error 141.334 px.

## Negative ablations retained

- signed lever-arm torque feature: worsened contact orientation top-1 to
  0.213 and motion top-1 to 0.577;
- RGB visual-contact gate: 4/64, mean XY 145.143 px, mean angle 1.629 rad;
- successor-orientation terminal phase: 3/64, mean XY 160.518 px, mean angle
  1.488 rad;
- geometry-state centroid interface: reduced mean XY to 121.122 px but only
  6/64 successes and mean angle 1.549 rad;
- fixed RGB torque servo: 0/64, showing that the remaining failure is not
  explained by the learned successor head alone.

## Current claim boundary

The repairs establish a reproducible improvement in held-out local action and
orientation diagnostics, and identify goal-angle observability plus long-horizon
contact control as the remaining bottlenecks. They do **not** support the claim
that the learned adapter stably solves PushT, supports reliable visual online
planning, or provides physical causal dynamics in general.

Artifacts:

- `contact_normal_dagger_temporaldynamics_softmode_explicitangle_historyreadout_h2_seed8411_20261010/causal_visual_world_model.json`
- `contact_normal_dagger_temporaldynamics_softmode_explicitangle_historyreadout_h2_seed9173_20261010/causal_visual_world_model.json`
- `contact_normal_dagger_temporaldynamics_softmode_explicitangle_historyreadout_h2_seed9173_20261010/mode_routing_audit/mode_routing_audit.json`
- `contact_normal_dagger_temporaldynamics_softmode_explicitangle_historyreadout_h2_seed9173_20261010/reversible_phase_contactservo_taskspace_eval20411_64/checkpoint_planning.json`
