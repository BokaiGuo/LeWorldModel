# K17 native capture storage seam

This prospective compatibility layer supplies `check`, `emit(kind, **fields)`,
and real PIL RGB224 PNG persistence over the independent 128 MiB bounded store.
An invalid capture array is encoded in the reserved failure record, subject to
the reserve limit; exceeding it fails rather than truncating evidence. The audit
function verifies the journal hash chain and saved fresh-observation file hashes
only. It does not decode pixels, authenticate a native source, check six-slot
ordering, or establish action/goal/early-stop correctness.

The journal schema intentionally differs from the older physical writer. An
actual adapter must consume this schema directly; it must not call an old
consumer or claim its status. No environment, model, or production run is
invoked here. Full runner, meter, failure lifecycle, source guard integration,
and full saved consumer remain required before MAIN production.
