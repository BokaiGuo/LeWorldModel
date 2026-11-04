# Wall-unicycle non-oracle diverse beam development test

After observing the original and injected results, compare the unchanged
depth-four Atlas search against a new non-oracle pruning alternative on
the same single exposed wall-unicycle task and three existing seeds. Both
use the original v2 candidate banks, scoring, root expansion, beam width 2,
full trajectory replay, and terminal selection. No waypoint candidate or
privileged successful trajectory enters either search.

At each intermediate cutoff, the new rule keeps the minimum current
task-cost node (stable candidate order for ties) and, among all remaining
nodes, the one maximizing the Euclidean distance of its final model state
from that best node after elementwise scaling by `[1,1,0.3]`; ties use
stable candidate order. The scale is a task-dependent development choice.
Unlike the old near-cost-tolerance variant, this rule has no cost threshold,
so it may retain a much worse near-term state and can hurt performance.

Implement this as a standalone search variant, with an accompanying
`cost` mode whose outputs, counters and complete programs must reproduce
the real unmodified Atlas search on the three seeds before interpreting the
`diverse` mode. Compare with original Atlas, full-horizon CEM and the
post-pilot random waypoint control. Count all backend transitions including
finite-difference internals, separate vector-distance arithmetic and
wall time; if candidate cardinalities change, costs may differ and no equal
budget claim follows. Save all complete programs, selected outcome,
counts and policy name. Tests must validate tie breaking and cost-mode
parity. This is one exposed analytical task and cannot establish RGB
closed-loop performance, reconfiguration mechanism or admission.
