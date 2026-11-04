# v13 Reacher RGB feedback: one-root development protocol

Test whether the exact v13 compiler used in the E0 development intervention
can run as a three-decision, fixed-terminal RGB controller on the frozen
Reacher model, and compare it with the same global-bank full-action CEM
control. This is a new task family, not a restart or replacement of the
stopped PushT contact root.

Use one explicitly fixed Reacher environment seed `20263271`, target delta
`(.65,-.65)`, action range `[-1,1]^2`, existing dataset-derived history
sampling and common executed warmup. Seed is not found by an exact text
search of the current `docs` and `experiments` trees, but broader history,
normalizer fitting and checkpoint training membership are unknown.
The goal image is produced by the environment wrapper. Feasibility within
15 steps is not certified; failure cannot be interpreted as a planner-only
failure or evidence of infeasibility.

Run v13 and a global-bank CEM once each from separately reset but identical
seed and warmup. For both use 15/10/5 remaining steps, execute five per
decision, re-encode real RGB feedback, and carry the actual unexecuted tail.
Global bank generation, seed, and incumbent/zero-first-packet final pair
match the previously tested JEPA interface. Each arm uses its own history
and carry. Each decision has a 1920 five-step prediction-token cap; record
full sample counts, SVD and elapsed costs separately. Same token counts are
not equal total computational budgets.

Save and hash all inputs, predictions, selected chains, dispatches,
observations, physical evaluator states and dependency receipts. Verify
common initial histories and goal, correct carry, candidate identity,
incumbent and pair, budget, and chronological decision/dispatch binding.
Run same-backend physical replay if available. The evaluator may read
physical qpos only for outcome reporting; the controller receives RGB,
goal image and action history. This single-process development adapter is
not a sandbox-based proof of information separation.

Execute each arm once and stop after all six planned decisions or a recorded
failure. No retries, root replacement, tuning, or post-outcome changes to
success or comparison. This is an interface and bounded physical control
diagnostic only; neither a result nor replay establishes PC2, PC3, PC4 or
PC5. Do not replace E3 contact or Cube evidence with Reacher.
