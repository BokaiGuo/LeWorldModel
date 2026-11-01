# P1 exploration source run 01

The presealed calibration-only request was executed before inspecting the
generated physical states. It completed 16 resets and eight explicit renders,
producing two selected source packs. This run made no neural-model calls.

The bounded follow-up receipt
`le-wm-main/results/p1_exploration_run01_receipt_20261101_hypatia_LOCAL02.json`
binds the request, source packs and journal, and reports no state7 or defined
pose-alias collision among the 16 recorded reset states against the covered
history and within the batch. This does not establish complete historical
coverage, development independence or training-data exclusion.

The source uses one observed RGB frame repeated as history with zero-action
padding. It is not a three-observation real history, and is not admitted as an
independent P1 confirmation source. Further real-history collection must have
its own fixed protocol and identity checks.

## Native angle termination hazard

Slot 1's goal angle is -1.559185605414262 while environment observations
return block angles modulo 2π. The installed `PushT.eval_state` computes
`min(abs(delta), 2π - abs(delta))` without reducing the difference first.
For the synthetic diagnostic of that goal angle, an observation angle of 6.0
and equal XY coordinates, the installed function returned native success,
although the correctly wrapped angle error is 1.276 radians (> π/9). This
is a code-level termination hazard, not an observed outcome from these roots.

Preserve and report the original native termination and any resulting
censoring, while separately recording the modulo-corrected angle and success.
Do not silently replace the benchmark gate or count a counterfactual
post-termination continuation as observed. Future execution must fail closed
if it requires uninterrupted physical data after such a termination.
