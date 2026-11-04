# AP-compatible PointMaze callback: CUDA smoke result

The [fixed development protocol](PC_AP_POINTMAZE_CALLBACK_SMOKE_PROTOCOL_20261103.md)
ran on exposed root 1888 with the frozen PointMaze checkpoint and 16 saved
action proposals. The new AP-compatible callback and the existing proprio
scorer produced identical costs for the same normalized five-step packets
(maximum absolute difference 0). Splitting the callback batch into two
halves changed scores by at most `3.725290298461914e-09`. The frozen
parameter hash was unchanged.

The callback evaluated 32 samples in three batches; the reference scored 16
additional samples. Two images were encoded. These counts describe this
smoke, not full planner cost. No physical trajectories, memory retrieval,
action selection, or AP baseline comparison were performed.

An initial run failed before scoring because `cuda` and `cuda:0` were
incorrectly compared as unequal device identifiers. The callback now checks
the requested device index when specified, then binds to the target tensor's
actual CUDA device. The same fixed inputs were rerun; no outcome was selected
or discarded. Analytic callback and existing kernel tests: **4 passed**.

Result [receipt](../le-wm-main/results/pc_ap_pointmaze_callback_smoke_20261103_DEV01.json)
SHA-256 `5f199e187d177339f0ab4cedcc32bb005f907aac4555187555a8e45fda034c55`.
The callback is an adapter for a one-packet predicted-proprio cost; it is not
the complete official AP controller or a reproduction of author results.
Retrieval memory, source-disjointness, real RGB feedback, physical execution,
and common total-resource accounting remain to be implemented and verified
for this PointMaze baseline.
