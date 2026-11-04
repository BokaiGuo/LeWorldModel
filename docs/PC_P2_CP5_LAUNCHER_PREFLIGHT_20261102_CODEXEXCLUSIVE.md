# P2 cp5 launcher preparation — blocked at winner authority

`p2_cp5_launcher_preflight_20261102_codexexclusive.py` supplies source and
native renderer checks, the existing physical module's exclusive Store/Meter
factory, and an fsynced single-process aggregate ledger wrapper. A caller must
freeze all resource caps before creating the ledger. Attempts, including failed
calls, count before dispatch; nested native calls are included when hooks use
the wrapped meter. Ledger hashes detect interior edits when checked against a
separately trusted tail; a standalone chain does not establish provenance or
prevent truncation.

The existing saved `multi_joint` result/consumer is a different, non-feedback
optimizer and cannot authorize a winner here. The feedback optimizer has no
established saved-output consumer in the inspected assets. The new winner
function therefore unconditionally rejects; it is not an executable physical
launcher. No environment or GPU is started by this module. The old prepared
physical source verifier and native environment validator are called by the
preflight API but have not been exercised against a real environment here.

Before enabling execution, implement a feedback-result consumer that verifies
frozen planning provenance, all 16 candidates per arm, model selection,
capture/encoded context and action bindings, report inventory and hashes;
bind its authoritative report and ledger tail to the launch contract. Add a
full launcher with shared-root reconstruction, fresh environment ownership,
cooperative locking, source rechecks, cleanup even on exhausted caps, total
disk/time safeguards, and saved-output verification. The present wrapper does
not itself create sessions, enforce an end-to-end wall/disk cap, authenticate
its caller's frozen limits, cover GPU/model/encoder work, or ensure successful
cleanup at a cap. These remain unmeasured or unimplemented, not zero.
