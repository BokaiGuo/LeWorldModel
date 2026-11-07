# Scored-incumbent real-policy parity protocol

On four already-exposed PushT evaluation tasks, run the installed closed-loop
evaluation twice with the same seed, action scaler, frozen LeWM checkpoint,
reset seed, horizon, solver seed, and search budget. Only the solver class
changes: historical scored-incumbent solver versus the selected-action-only
output adapter. Save each run's full physical trace, solver audit and metrics
separately. The policy uses RGB and goal images; the evaluator uses simulator
state to initialize and assess tasks. This protocol does not establish strict
process isolation, training-set independence, or RGB provenance.

After both executions, compare ordered solver audit records except elapsed
time, selected actions and costs, and physical traces sorted by reset seed and
step. Require identical candidate-selection metadata and actions, and physical
states within absolute tolerance 1e-5. Checkpoint and normalizer files are not
modified. Report termination and total model score calls, and retain all
mismatches. A small exposed parity test is not a quality comparison, an
independent evaluation, or scientific gate passage. If a run fails, preserve
its partial output rather than overwriting it.
