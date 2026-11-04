# v13 PushT new-root contact witness: fixed development protocol

To avoid reopening the stopped v12 contact root, use exactly one new native
PushT seed `2026110433`, the established ten-action warmup, and the
evaluation-only feedback law `float32(clip((object_xy-agent_xy)/100,-1,1))`
for 15 actions. Replay the same actions once from reset and require exact
RGB, proprioception, and physical state at all 26 boundaries. A nontrivial
goal requires object-centre displacement greater than 20 pixels or wrapped
rotation greater than π/9. Record failure and stop without another seed if
the law, termination, replay, or nontriviality condition fails.

Only if this independent new-root witness exists may its final RGB be used
as the goal for the separately fixed v13 feedback protocol. Witness actions,
privileged state, and construction process are evaluator-only; they cannot
enter planner initialization or the model-visible sandbox. This constructed
goal is not an independent benchmark or method performance.
