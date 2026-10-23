# PointMaze strict oracle protocol audit

## Finding

The earlier strict PointMaze comparison used the labels
`ground_truth_candidate_oracle` and `perfect_dynamics_mpc` as if they were
terminal-task ceilings.  They were not.  The first exactly ranks one block of
the fixed action bank against the current RGB-derived waypoint; the second is
a depth-limited, width-limited beam search against that same waypoint.  The
reported outcome, however, is terminal distance and success at the environment
goal after re-planning.

Consequently, an RGB controller can have a better terminal result than either
privileged control without simulator-state leakage or a contradiction.  This
is an objective--horizon mismatch, not evidence that a learned controller has
beaten a global physical oracle.

## Repairs

The strict runner now requires all six arms to bind:

- one root-fixed exact candidate-action bank and every executed action's bank
  membership;
- terminal evaluation objective
  `terminal_global_goal_distance_v1` and one-block re-planning semantics;
- the canonical RGB-calibrated planner goal qpos, the evaluator goal qpos,
  their measured gap, and target chart;
- `one_step_active_waypoint_distance_v1` plus exact one-step scope for the
  local privileged ranking control; and
- `bounded_beam_active_waypoint_distance_v1` plus non-complete-search scope
  for the finite-horizon privileged control.

The candidate bank, active probes, RGB route construction, and local oracle
now share the same per-episode RGB-calibrated planner-goal qpos.  The raw
environment goal remains the terminal evaluation target.  This repairs a real
target-identity discrepancy: the two qpos targets are close but can differ by
more than a candidate-ranking margin.

## Fresh GPU artifact

`results/pointmaze_rgb_closed_loop_strict_20261021_cuda_full_d2b8_objective_contract/`
was regenerated after these contracts were frozen.  It completed the six-arm,
12-root GPU panel and passed the strict validator.  On this one panel, the
ungated RGB effect adapter has mean terminal distance `0.18348` versus
`0.25207` for frozen visual dynamics; its paired 95% bootstrap interval is
`[-0.12993, -0.01465]`.  This is a held-out same-dynamics RGB control signal,
not an authorized HTSC result: the HTSC arm's fallback reduces the effect
adapter's success to `10/12` and its consistency-gated arm to `9/12`.

The local one-step privileged control and bounded beam control have terminal
success `9/12` and `10/12`, respectively.  They are diagnostic comparators,
not reachability ceilings, and must not be used for P0 causal routing.

## Remaining boundary

The current evaluator goal lies close to a legacy XML wall boundary.  A
separate goal-region and route-reachability audit is required before claiming
that terminal Euclidean distance to this point is a robust global planning
objective.  Until then, the artifact supports only the declared fixed-bank,
same-dynamics closed-loop comparison.
