# Non-progress prefixes: feedback replanning and cross-suffix result

The [protocol](PC_POINTMAZE_NONPROGRESS_FOLLOWUP_PROTOCOL_20261103.md)
completed 12 real-RGB feedback trajectories and 36 fixed-action cross-suffix
position evaluations on four exposed roots. Prefix terminal RGB matched the
sealed bank, all 12 full trajectories matched fresh-bridge terminal RGB,
and the selected/greedy diagonal endpoint positions matched the original
feedback trajectories. Each conditional suffix planner used four decisions
and 50 predicted candidates per decision.

| Root | Selected cp5 distance | Selected replanned endpoint | Same-bank greedy endpoint | Zero endpoint |
| --- | ---: | ---: | ---: | ---: |
| 8343 | 1.487012 | 0.410741 | 0.119578 | 0.244271 |
| 8345 | 2.258941 | 0.515513 | 0.039219 | 0.504585 |
| 8349 | 1.604257 | 0.119734 | 0.476921 | 0.015041 |
| 8351 | 2.706745 | 0.786514 | 0.102364 | 0.606551 |

The selected prefix ends closer than greedy on 1/4 roots and closer than
zero on 0/4 roots under this common conditional optimizer. It did not
establish the required planning advantage. Each selected prefix had no
initial goal-distance improvement, but that alone does not establish a
useful reconfiguration.

The selected-optimized common suffix had a different relative-to-zero
effect across the three prefixes; e.g. on root 8349 it changed terminal
distance by -1.302819 after selected and +0.210716 after zero. This is an
action-response interaction, not a capability advantage: that suffix was
optimized from selected, and prefix position and velocity differ. Suffix
replanning, cross-suffix intervention, and full-method equal-resource
comparison are different experiments. Roots and outcomes were exposed,
and the model-only selector uses no future-value criterion. No independent
confirmation, strong-baseline win, or P1 pass is established.

Receipt [DEV01](../le-wm-main/results/pc_pointmaze_nonprogress_followup_20261103_DEV01.json)
SHA-256 `99fee8b199b822c6b9eb833bf4631a1f1d3d2c07a77a71723f1f58472437e922`.
