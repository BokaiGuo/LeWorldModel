# v25 PointMaze RGB development feedback contract

This wrapper applies the same v25 response configuration as its E0
development panel (`max_modes=2`, response retention) and the shared-bank
CEM baseline to the existing fixed-terminal PointMaze RGB harness.
Each decision has a 1,920 prediction-token cap and plans the remaining
15, 10 or 5 physical steps. Both arms receive the same exposed target,
legal RGB-derived context, action history, previous-plan tail and paired
abstention rule. Actual dispatch is five steps followed by fresh RGB
and replanning. The harness seals actions before dispatch and records
its model, context, action, timing and execution dependencies.

This is a development integration contract, not a confirmation protocol.
An actual run requires a separate immutable development task contract with
new output name, legal initial state and explicit known overlap checks;
do not reopen stopped roots. Checkpoint-training and normalizer-source
independence remain unknown. Existing RGB scores do not establish that
the task is physically solvable or that the model score ranks physical
outcomes correctly. Report physical observations and all failures; do not
substitute model score for success, and do not claim PC4 or PC5 from a
single run. Prediction-token accounting is not full encoding/GPU cost.
