# K17 first5 transport seam (not a production adapter)

Exploratory candidate: all six frozen K17 slots, each reset and warm10 then
literal first5 of the existing model plan. This is not a full-plan deployment,
not a physical receipt, and not an equal-compute claim. No dispatch is exposed.

The integer census takes 306 PNGs: six slots times three renders times
(11 warm frames plus six plan frames). Its conservative PNG formula is copied
from the accepted fixed-size 224 RGB census. The remaining 64 MiB after 8 MiB
reserved and the PNG upper bound is only an aggregate metadata allowance; it
does not prove journal, report, completion and failure evidence fit. Actual
four-slot Z/P journal and report sizes are observational only, not bounds.

The exclusive journal provides canonical one-newline records, chained hashes,
fsync, a pre-write byte limit and fail-closed error handling. It is a fixture
seam only: it does not count PNGs, enforce disk floor/wall, publish failure
reserve, or certify source identity. `read_journal` is not a full consumer.
An incomplete write is retained and must be rejected, not repaired.

Outstanding before any production: bind and verify frozen replay bridge and
live guard to actual reset/warm/goal capture; budget every actual operation;
implement bounded total writer and durable early-stop/failure publication;
prove metadata capacity with all copies; fully validate native events and PNGs
and all six slots in a separate saved consumer. MAIN must review and authorize
the full contract. Nothing here invokes physics or a model.
