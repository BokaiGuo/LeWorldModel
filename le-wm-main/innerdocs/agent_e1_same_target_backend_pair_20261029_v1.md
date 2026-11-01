# E1 actual backend pair integration slice

`run_pair` executes both independent original 72-candidate controllers with
native `beam.Dispatch`, explicit caller-supplied backend guards and the fixed
actual-hook meter, seals and reloads each controller, then compares complete
bank bytes before either extra candidate. It runs the fixed extra H2 and
selected whole/1+1 replay through the same Dispatch and guarded real hooks.
Per-phase actual attempted/completed counters are copied in `finally` even on
failure. `verify_saved` independently replays both saved controller IO via
the existing strict controller reconstruction, reconstructs the extra edge
with `SavedDispatch`, recomputes original Q and 73-way tie break, and checks
whole/segmented saved replay with the existing verifier.

This is not yet an executable production runner: the caller must implement
durable hash-chained sinks, an exclusive bounded seal/reload store and failure
report in its own `finally`, bind authenticated `source_inputs()` to the actual
loaded task/model and full source closure, enforce CUDA-only model and before/
after weights, all-four-slot/global seal barriers, token/raw identity, resource
census, complete saved consumer and conservative output/memory/wall budgets.
In particular the current API handles a root pair, not the global four-slot
barrier, and does not supply a MAIN run command. No readiness or native
reachability is asserted; main authorization and the frozen protocol apply.

Tests use a non-neural fake backend with the real Dispatch, actual-hook meter,
temporary journal and exclusive seal/reload, including tamper rejection. They
do not execute GPU, native physics, training or an upstream fullstrict.
