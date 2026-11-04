# Goal-relevant modes: isolated compiler integration

This bounded side task does not alter default compilers, stopped PointMaze
constructions, historical gates, or any main-thread result. It uses the
existing exact-dynamics backend only, not CPU inference of a visual model.

First revalidate the completed response-program joint-refinement result:
source/protocol hashes, all selected actions by replay, summary distances,
budget arithmetic, and the recorded final-chain intervention predicates.
Do not rerun or overwrite that experiment.

Implement an opt-in compiler using the existing goal-mode ranking function.
For K = sqrt(Q) G inverse_sqrt(R), compare:

* task: retain at most r nonzero modes by
  q_i = s_i^2/(s_i^2+lambda) * (u_i.T residual)^2;
* gain: retain at most r nonzero modes in singular-value order.

Use precisely those retained modes for the progress update, bridge directions
and local ridge reachability value. A zero response yields no identified mode.
Both rules compute the same decomposition and contributions once. Keep the
existing bounds, nonlinear line search and explicit action witnesses.
Final plan selection uses the existing Atlas complete-chain cost. The opt-in
compiler exposes compile(context), so the existing generic feedback adapter
can use it; that interface test is not a visual/physical experiment.

Before execution fix 16 already exposed radius-.9 goals at
2*pi*(i+.37)/16. For both arms: original unicycle backend, identity action
scale, position-only task metric, one-step packets, one retained mode,
Atlas depth 2, beam width 2, two random bridge proposals, seed 2026111200+i,
rollout cap 1000 and Jacobian cap 100. No outcome-dependent tuning.
Record all programs, selected index, separately qualified index, selected
chain equality, mode ranks, singular values, task contributions, actual
rollout/Jacobian/SVD calls, action steps and elapsed planning time.

The zero-action nominal of this backend may yield only rank-one task
response. If so, there is no second nonzero mode to reorder, and this panel
cannot identify a benefit of task ranking. Do not change the nominal or task
after seeing this result. A rank-two linear fixture verifies that an
irrelevant high-gain direction loses to a relevant lower-gain direction; it
is a software test, not research evidence.

Preserve complete pre-refinement/joint-refinement outcomes from the other
experiment. This task does not introduce new refinement, new roots, a trained
score, privileged information into an RGB planner, or a new admission rule.
All data here are development or fixture evidence. No P1/PC gate can open.
