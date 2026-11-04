# Prefix-feedback capture/archive bridge

The new bridge connects per-candidate `NativeSession` output to the existing
feedback kernel ABI and structural saved-output consumer without modifying old
receipts. It durably records capture intent, actual RGB bytes, normalized action
history, snapshot, explicit returned stop flags, context, native journal and
model events. A complete seal requires all 32 ordered captures and contexts,
source-role files and the aggregate budget ledger; it invokes the independent
consumer before returning the new manifest digest. Failures are retained and do
not produce a successful receipt. Stops never produce padded capture data.

The end-to-end test uses a stateful fake physical provider and fake optimizer
callbacks; it calls no actual model or simulator. `Journal` in the new module
is a fake-test store, not the historic physical store or proof of a prefix.
The bridge can copy the historic store's journal into the flat archive, but
this path and the real physical hooks have not been exercised.

The bridge takes a session factory and input source references from its caller.
It does not authenticate them, acquire a cooperative lock, create or close real
environments, enforce disk/wall caps, or authorize a winner replay. The
consumer's receipt explicitly keeps source and physical authentication and
physical authorization false. Further, the consumer cannot independently
recompute unsaved gradients or model outputs. Missing or unverifiable sources
remain a blocker, not an authorization to execute.
