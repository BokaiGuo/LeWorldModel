# Prefix-feedback saved-output consumer v1

This is an independent, CPU-only structural consumer for a **new** archive
format; it does not consume the unrelated historical multi-joint result. Call
`consume(archive_directory, externally_pinned_manifest_sha256)` and retain the
returned receipt. The manifest binds source artifacts for preparation, task and
bank, model, normalization, code, environment and device/lock; no role may be
absent or empty. Files are local, flat, non-symlinked and SHA-256 checked.

The archive must include the 16-entry raw float32 bank in fixed order, full
kernel result and events, each arm/candidate's receipt, actual RGB byte tensor,
encoded context and native session journal, plus aggregate budget journal.
Each receipt must explicitly record both native stop flags. Current prototype
capture receipts do **not** contain those flags and therefore cannot yet
produce a compatible archive; the future producer must add evidence in new
files, not reinterpret missing flags as false. The manifest pins warmup and
shared-prefix actions. Each native journal must show the root guard, warm10,
guarded shared15, candidate5, no stop, and matching cp5 snapshot/RGB.

The consumer checks every candidate/arm optimizer's nine returned scores,
incumbent and tie rule, bank and prefix invariants, model resource totals,
32 capture/context bindings, candidate-index selection and winning raw action.
The single-root capture-only physical ledger must have exactly 30 public
steps for each named `capture_{index:02d}_{arm}` session (960 total), no failed
or unmatched calls, and a tail matching the externally anchored manifest.
Other resource kinds require explicit counts; budgets for later winner replay
are not included. The producer must independently pin the manifest and ledger
tail before treating either as evidence.

The consumer verifies byte and event consistency, not that a source file is
the genuine checkpoint, environment, lock, physical capture or CUDA process.
Those role-specific attestations and final source rechecks remain unavailable.
The current optimizer does not persist gradients, so gradient updates and
model scores are not recomputed. A self-consistent fabricated archive can
pass structural checks. The returned `physical_authorization` is always false,
as are source/physical authentication and gradient-recomputation flags.
No result here establishes equal total computation, historical SRA, independent
confirmation, or P2 admission.
