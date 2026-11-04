# Trace injected wall-unicycle Atlas beam

Replay the three existing privileged waypoint-chain injection runs without
changing the search or candidate banks. Instrument the actual v2 search
immediately after its beam assignment, only for root branch `injected_0`.
Record each expanded node's edge names, full context digest, current model
cost, stable rank and actual retention, and verify the observed beam equals
the actual stable cost-sorted width-two slice. Trace all three suffix depths.

Compare the exact named two-packet witness against its independently bound
successor context and record whether it survived. Confirm final selected
actions, backend counts, and offers match the previous injection receipt.
Constructed future chain is privileged; the trace is not a planner
performance measurement and cannot establish P1. Runtime hooks must fail
closed if the expected assignment cannot be found or another trace is active.
Score calculations performed only for tracing are extra unmetered arithmetic;
compiler transition counts are reported separately.
