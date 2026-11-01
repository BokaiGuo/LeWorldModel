# P3 resolved allocation physical replay protocol

Fix the complete Cartesian roster before inspecting this experiment's model
results: points 20 and 30, both known roots, all three arms uniform, signal
and reverse; six branches per point, twelve total. Execute each branch once,
including repeated actions; no cherry-picking by model or physical outcome.
Actions are each arm/root's selected winner from the corresponding newly
saved model run and are bound by raw float32 hashes to fixed root, task and
prior native source. Preserve native stops and partial failure. No model
inference occurs during this replay.

Only run after all six model slots were delivered, consumed and showed
different signal/reverse candidate allocations for at least one budget point.
Rebuild each model consumer in a fresh CPU process and verify all hashes
before deriving the physical prepare. Use exclusive new output paths and the
existing physical operation caps/guards. One point's failure does not make
the other point's existing data successful, nor does any incomplete branch
count as a completed endpoint.

This protocol is after older known-root results, and physical prepare follows
new model execution. It is an exploratory fixed-roster replay, not an
independent or blind confirmation, equal-total-compute comparison, P1 pass or
P3 scientific admission. Physical and neural costs remain separate.
