# Saved suffix-expression collision-branch diagnostic

Analyze only already saved batches from the stopped four-root suffix
expression panel. No planning, optimization, new candidate, or simulator
search is allowed. Verify the source result, source audit, and all trace
hashes. For each fixed-prefix suffix arm (four- versus ten-knot CEM), root,
and candidate, reconstruct saved trajectories using the existing alternate
same-environment line-segment geometry.

Report counts of saved trajectories, trajectories with a blocked step,
trajectories ever reaching `x > 0`, their minimum recorded terminal
distance, and success below the unchanged 0.05 threshold. Treat the
zero-suffix initial evaluations separately. Bind each batch to its saved
candidate/run and validate the count and actions. Check that all saved suffix
batches are accounted for exactly once. Count roots as task units; optimizer
samples are not independent tasks. Also report the number of prefixes that
ever obtain a crossing in each arm. Use no data-dependent threshold or
new stopping rule.

The geometry is another implementation of the same analytical dynamics,
not a real-world replication. A lack of crossings in a finite bank cannot
prove that a task or prefix is infeasible. This does not modify the original
panel, add optimizer opportunity, establish a bug, or change a PC gate.
