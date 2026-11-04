# P2 cp5 split binding prototype — development only

This is a fake-session-tested binding core, not a physical adapter or launcher.
Do not use it to execute a real environment. It does not verify that the
session provider's root, warmup, shared prefix, observations, snapshots or
guard are genuine. In particular, a nonempty guard string is not evidence.

The callable capture ABI matches the existing feedback kernel. Candidate raw
float32 bytes, RGB bytes and normalized action history are recorded; a winner
replay reconstructs a fresh provider session, checks the prefix, RGB hashes,
history, provider guard and snapshot before dispatching its suffix. Capture
and replay failures are retained and stops are not padded. Model winner
authenticity and the fixed candidate/arm ordering are not established here.
The prototype accepts a winner supplied by its caller, and therefore cannot
rule out outcome-based selection. It is not a saved-output consumer.

Before any actual execution, implement and independently test a native
provider which reproduces and journals original placement, warm10, root and
goal, exact shared-prefix events, and cp5 state; ensure per-operation durable
failure/stop records, independent saved-output revalidation and frozen
optimizer-result selection. Enforce candidate/arm ordering and complete
denominators, per-root and per-run limits, cooperative locks, source and
environment identity, and prohibit dispatch on any binding failure. The
prototype's counter covers only attempted candidate/replay public steps;
warmup/shared prefix, native physics substeps, rendering, encoding, GPU/model
work, wall time and FLOPs are not measured. Its 170-step local cap is not a
full physical resource cap. No claim of equal total cost, historical SRA,
independent confirmation or P2 admission follows.
