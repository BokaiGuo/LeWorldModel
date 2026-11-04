# Exact unicycle: oracle and multistart gradient development comparison

Reuse the 64 deterministic, already-observed angular goals and the original
two-step exact unicycle, initial pose and action bounds. The panel is not an
independent task sample. Preserve the previous pilot and its scores.

Add (1) a task-specific analytical reachable witness: choose first-step
velocity `x_goal`, choose a turn of `+pi/2` or `-pi/2` according to the sign
of `y_goal`, then choose second-step velocity `abs(y_goal)`; (2) deterministic
multistart L-BFGS-B over the effective first velocity, first turn and second
velocity, with analytical objective gradients and at most 32 combined
objective/gradient evaluations per task. Four fixed starts are specified by
the implementation; take the best observed point, including when the budget
terminates an optimizer. The second angular action is irrelevant to terminal
position and fixed to zero.

Also preserve the previous fixed five-turn bridge, greedy, random-32 and
CEM-32 results from their sealed report, by root index and goal. Track
objective/gradient evaluations, replay transitions and analytical response
operations separately. The oracle is problem-specific and not a generic
planning method; gradient computation is not equated to a forward rollout.
Do not claim equal wall time, generic planner supremacy, learned-model
mechanism, or independent generalization. Validate analytic gradients by
finite differences and validate the oracle action witness against dynamics.
