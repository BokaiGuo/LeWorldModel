# Constructor close guard: real PushT development smoke

The old `metered_class` catches constructor errors and invokes `self.close()`
directly. The exclusive adapter subclasses the actual metered class and routes
that virtual call through `meter.call("close", ...)`. It tracks an outer
close callback so a normal host-style `meter.call("close", env.close)` is not
double charged. This does not modify the old class, source, or host gate.

The fixed smoke uses the source-hash-pinned real installed PushT and registered
Gym lineage, CPU physics only, and independent capped journals for two cases:

1. One constructor, seeded reset, render, one zero-action public step, close.
2. An explicitly injected error immediately after real PushT constructor
   returns. The old constructor handler must invoke the adapted close; the
   expected constructor failure remains failed, not a successful episode.

Each case has a 30-second cooperative check and the fixed independent resource
vector in the script; a process-level timeout is also required. The prepare
file binds the adapter, physical guard, legacy meter/moving sources, native
PushT file, smoke code, tests, and this document. It does not bind the complete
Python, SDL, Gym, NumPy or physics dependency closure. Native bytes and other
sources are checked again before the run; no existing output may be reused.

These are development diagnostics, not paired P1 trials or a complete
physical-cost certification. The adapter covers close via the subclass's
virtual dispatch only. Deliberate base-class calls, captured methods, other
threads, external resources, visual-Sink composition, upstream initialization,
and total-cost completeness remain unverified. The existing full-host gate
remains closed; no old result or contract is changed.
