# Exact unicycle planning development protocol

Purpose: test whether a short, no-immediate-position-progress action can alter
the task-relevant response of the same subsequent action channel under exact
known dynamics. This synthetic control experiment is not evidence about the
frozen visual model or real robotic control.

State `(x, y, theta)` evolves with action `(v, omega)` as
`(x + v cos(theta), y + v sin(theta), theta + omega)`; bounds are
`v in [-1, 1]`, `omega in [-pi/2, pi/2]`, two steps. Objective is terminal
Euclidean position error to a goal at unit radius. Use 64 evenly spaced goal
directions offset from the grid by 0.031 radians; initial pose is zero.
All tasks and seeds are fixed before execution. This panel is synthetic and
constructed for development, not independent real-world confirmation.

Compare four algorithms:

1. one-step greedy: analytically choose bounded immediate velocity, tie-break
   angular action to zero, then repeat;
2. full-sequence random search, 32 sampled sequences per root;
3. full-sequence CEM, 4 iterations x 8 sampled sequences per root;
4. structured two-stage: evaluate five turn-only prefixes at angles
   `[-pi/2, -pi/4, 0, pi/4, pi/2]`, then use each successor's exact local
   velocity response to analytically choose its bounded progress action;
   select the best full program. The prefix family has no position progress.

All algorithms use the exact same dynamics, horizon, action bounds and
terminal metric. Track actual dynamics transitions, objective evaluations,
and analytic response/Jacobian evaluations separately; never equate unlike
operations or claim equal wall-clock. Random/CEM receive the larger limit of
64 dynamics transitions. Use per-root deterministic generators independent
of task outcome. Record every action witness, first-step and final distances,
position-response vector and successor response vector. The zero-turn prefix
is the paired reference for capability change.

Report every root, mean error, exact wins/ties/losses and mechanism witnesses.
No confidence interval or significance claim is based on the angular grid.
The mechanism requires both zero immediate improvement and a changed
goal-directed response with better final error than zero-turn plus its own
optimal second step. A result better than a weak baseline does not prove
advantage over gradient MPC, GN, GRASP or a learned world model.
