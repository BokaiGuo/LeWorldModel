# K17 native six-slot outer core (not production-ready)

`material` invokes the pinned paid-replay bridge for all six fixed identities,
loads the original native sources, and compares reset identity, warm trace and
root hash. `execute_material` is a six-slot execution core using the new meter,
native episode and bounded store API; native early stops retain the row and
continue, while engineering errors leave later rows `not_started`. Fixtures
use fake environments only and establish no physical provenance.

This is not yet a runnable production adapter: prepare and source-closure
sealing, durable report/completion/failure publication, inclusive wall
certification and an independent complete saved consumer are absent. Calling
the core directly bypasses those prerequisites. It does not confer physical
release or equal-budget claims. No production CLI or receipt is provided.
