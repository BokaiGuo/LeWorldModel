# E1 four-slot durable staging bridge

The bridge routes the fixed four-slot orchestration into the existing bounded
E1 `Store`: one hash-chained ordered event stream, exclusive compressed payloads,
and immediate reload checks for every controller, choice and replay. It returns
all four row states and resource prefixes even after a dispatch exception, and
attempts a failure event. `finish` closes the journal and publishes a bounded
report, completion and timing witness; the owning runner must call it in its
own `finally` block. On journal, disk or
fsync failure neither a complete report nor an emergency marker is guaranteed.

`verify_payload` reconstructs all four saved controllers/extra edges/replays,
checks slot payload identity, full 144/2/4 module counts and samples and the
global timeline. It assumes the caller has first checked the report SHA, every
artifact SHA, compression/array bounds and hash chain with `old.read_store`,
and authenticated the actual task/source/model authorities. Partial reports
are diagnostic only and cannot become a successful receipt.

This remains fixture-only. No full production runner/CLI, model loading,
before/after weights guard, task authority, source closure, 4-slot event/byte
census or fully independent emergency writer is implemented here. Existing Store
limits are fail-closed and not proof that this workload fits; do not start
production. No NN/physics or upstream fullstrict was executed.
