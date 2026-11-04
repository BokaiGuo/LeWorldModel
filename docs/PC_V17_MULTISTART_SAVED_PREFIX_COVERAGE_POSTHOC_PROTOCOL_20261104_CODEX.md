# Saved-prefix coverage: post-hoc diagnostic protocol

Use only saved results of the stopped v17 versus multistart E0 panel.
Select every task on which the saved multistart CEM final chain succeeded
and v17 did not; the selection is explicitly outcome-dependent. For each
such task use the CEM selected chain as a diagnostic reference, never as
an input to a planner or as a method score.

At each recorded v17 search depth compare RMS action distance from the
reference prefix to all generated and all retained prefixes. Record the
nearest candidate, its retained status and scores, plus the v17 final-chain
distance from the reference. Verify the source plan/audit hashes, retained
indices and that the recorded CEM reference was scored. This is offline
trace inspection; do not evaluate new trajectories, optimize or rerun
the stopped panel. Action-space distance alone is not a measure of
reachability, utility, or a causal diagnosis.
