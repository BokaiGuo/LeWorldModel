# Native endpoint replay for history CEM development trajectories

Replay all 16 sealed trajectories in
`a1_pointmaze_rgb_history_cem_feedback_20261103_DEV01` with the legacy
MuJoCo RGB backend used by the actual executor and separately with the
position backend. Use the saved prefix and selected tail, five high-level
actions each; initialize from the same archive float32 state. Recompute and
require cp5/cp10 RGB hashes to match the original actual executions.
Require the position backend to reproduce its stored endpoint exactly.

Evaluate the final legacy replay state using the checked-in
`PointMazeWrapper.eval_state`: success iff position distance is strictly
less than 0.5, and additionally report its four-dimensional state distance.
The goal is the same archive frame-10 state used to supply the goal RGB.
This is deterministic **post-outcome replay**, not a direct observation
of hidden simulator state in the original bridge. Record legacy-to-position
differences, backend/runtime/XML/source hashes and every arm.

The native maze environment's sparse reward uses `<= 0.5`, whereas the
planning wrapper's `eval_state` uses `< 0.5`; report the latter explicitly.
The original ten actions ran without success-based early termination and
the wrapper's offline `eval_state` does not imply stopping. The exposed
roots, fixed prefixes, unknown checkpoint training provenance, mismatched
total planner cost and absence of strong baselines exclude confirmation or
P1 claims. No simulator state is sent back to a planner.
