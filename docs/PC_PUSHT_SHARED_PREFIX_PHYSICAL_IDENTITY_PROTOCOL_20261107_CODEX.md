# Shared-prefix physical identity audit

Use the four previously captured development tasks and all 301 shared-prefix
chains per task. Verify candidate action fingerprints and exact equality of
the first two normalized five-step action packets, then match each chain to
its previously recorded physical replay. Compare its recorded observable
physical state at steps 5 and 10 with the selected chain and the selected
chain's recorded online trace. Also reconstruct the selected chain's first ten
environment actions using the dataset-fitted scaler and compare them to the
actual trace. Report maximum absolute discrepancies and counts exceeding
1e-4 for states and 1e-6 for actions.

This is an audit of existing recorded replays, not an independent physical run.
Recorded state is not a complete simulator snapshot; equality at packet
boundaries does not prove equality of hidden contact or full internal state.
No physical state or success label is made available to the planner. The
panel was selected previously and cannot be used as independent confirmation.
