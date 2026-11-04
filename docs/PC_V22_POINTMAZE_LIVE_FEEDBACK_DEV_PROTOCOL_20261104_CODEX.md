# v22 PointMaze live-feedback development diagnostic

Freeze before execution. One paired development setup, start `(0.85, 0.85)`,
zero velocity and the existing obstacle goal image. This nearby new setup
does not establish a previously unseen distribution, checkpoint independence
or normalizer independence; all are unknown. It is not a confirmation panel.
The exposed goal and calibration come from the existing PointMaze pipeline.

Run response v22 and multistart CEM, both with a 1920 prediction-token cap
per decision, the same frozen encoder/predictor, goal, normalizer, action bounds,
warmup and three real feedback decisions. Plan horizons are 15, 10, 5 physical
steps, and each decision dispatches five. A carried tail is rescored at the
new observation and made available to both arms. State is used only to
initialize the isolated renderer; the planner receives RGB-derived estimates
and past actions, not simulator state. This is not a claim of RGB-only
benchmark certification: the existing calibrated visual estimator must be
separately audited.

Save dependencies and hashes before execution, every scoring request and
result, observed frames, context/action-history bindings, the full selected
chain, carry and resources. Persist each decision before dispatch and verify
paired warmup images match. Refuse to overwrite any output or retry an
incomplete run under the same name. Report failures and use one run only;
no replacement roots, tuning or extended horizon in this protocol.

The bridge returns RGB, not physical state. This diagnostic by itself cannot
establish physical success, fairness of total compute, PC4 or PC5. Prediction
tokens are a partial resource axis; timing and encoding counts are recorded
but full GPU and decomposition costs are not certified. A separately sealed
physical replay may evaluate saved actions after execution without feeding
results back into the planner.
