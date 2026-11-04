# CEM incumbent physical development protocol

For each of the two known development roots, execute the first packet of the
incumbent-retention replay plan, then replan the final two packets from actual
RGB and executed history using the same suffix CEM and seed as the earlier
full-CEM run. The original 192 initial candidate rollouts are accounted from
the sealed model replay; their wall time is not included in the physical
executor timing. No new first-stage search or outcome-based plan selection is
allowed. Bind model replay, original CEM and root reports by SHA and verify
root/goal pixels, checkpoint and action history before execution.

The same old roots and observed results make this an exploratory execution,
not confirmation. It tests whether keeping the best prior CEM sample affects
physical outcome, not whether lower predicted cost ensures better control.
