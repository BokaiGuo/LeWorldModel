# Fixed-action persistent backend comparison

The eight previously executed 3.1.6 trajectories were replayed action by
action in evaluator-only MuJoCo 2.1.2 using the official shadow-executor
XML and in MuJoCo 3.1.6 using the original diagnostic XML. The 3.1.6 replay
reproduced every saved position and velocity exactly. Maximum discrepancy
between the two backends was 0.224477 in position and 0.347095 in velocity;
two action boundaries disagreed on particle-wall contact presence. Clocks
agreed exactly. The fixed-panel consistency rule therefore failed.

| Start | Policy | Recorded 3.1.6 steps | First recorded wall contact | Maximum position discrepancy |
| --- | --- | ---: | ---: | ---: |
| (0.8, 0.8) | direct | 22 | 5 | 0.224477 |
| (0.8, 0.8) | waypoint | 15 | 12 | 0.012038 |
| (0.8, 1.8) | direct | 21 | 5 | 0.047794 |
| (0.8, 1.8) | waypoint | 11 | 8 | 0.007679 |
| (2.8, 0.8) | direct | 22 | 5 | 0.224477 |
| (2.8, 0.8) | waypoint | 15 | 12 | 0.012038 |
| (2.8, 1.8) | direct | 21 | 5 | 0.047794 |
| (2.8, 1.8) | waypoint | 11 | 8 | 0.007679 |

Contact counts refer to action-boundary observations, not the first
integration substep with contact. Some divergence was already visible at
the first recorded contact. Full state, contact pairs and per-step errors
are in the [receipt](../le-wm-main/results/pc_geometry_backend_stepwise_20261103_codex.json).

The follow-up [2×2 factorial](../le-wm-main/results/pc_geometry_backend_factorial_20261103_codex.json)
replayed both XMLs under each engine. Within either engine the two XMLs
produced identical recorded states and wall-contact indicators. With the
same XML, the engine-version difference remained 0.224477 / 0.347095 with
two contact-indicator disagreements. This isolates the discrepancy to the
specific runtime change rather than this XML substitution for these
trajectories; it does not isolate a particular solver cause.

The 3.1.6 waypoint traces first reach the pre-existing distance threshold at
steps 15, 11, 15 and 11. On the 2.1.2 replay the same waypoint actions also
reach it at those steps. The corresponding direct action traces were
stopped by the original 3.1.6 run at steps 22, 21, 22 and 21; none had
reached the threshold on 2.1.2 by the end of the recorded actions. **They
are right-censored at those recorded endpoints**, not proven failures at
the 24-action cap. The stepwise protocol's proposed 24-step failure
classification cannot be applied without executing additional, separately
defined actions; no such extension has been performed.

These are exposed, privileged reference traces, not a comparison of RGB
planners. Greedy reaching a target does not rule out an efficiency
advantage, and a faster privileged waypoint is not evidence of a
dynamics-program advantage. The 3.1.6 experiment remains a separate
environment protocol and must not be pooled with formal 2.1.2 outcomes.
Four targeted tests verified the source binding, action replay, per-step
differences and factorial comparisons.
