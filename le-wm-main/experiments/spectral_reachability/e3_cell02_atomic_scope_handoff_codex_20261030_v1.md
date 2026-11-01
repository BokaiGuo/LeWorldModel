# E3 cell02: atomic scope and manifest validation handoff

This is preproduction validation only. It does not launch, reserve resources,
claim cell02, approve a manifest, or change the fixed 16-cell scientific
contract. The existing cell02 claim remains independent; cell03 is out of scope.

The read-only validator accepts only the existing final-review manifest path
policy and an externally supplied SHA-256. It compares the approved canonical
document to a fresh candidate draft from the existing isolated subprocess,
with the inherited environment unchanged, then checks manifest, validator,
authority, interpreter and environment again. This is a point-in-time check:
it does not bind later execution, rule out transient changes, attest native
libraries, verify GPU availability, or replace the existing resource review.
An independently reviewed manifest must be generated in the actual final
environment and externally pinned; test fixtures are not approval.

The temporary-only claim primitive demonstrates `O_EXCL` competition on one
fixed cell02 marker across cooperating processes, independent of entrypoint,
manifest digest and output label. It never removes a marker, including on
partial writes or fsync errors. Existing malformed files and symlinks block.
Tests use temporary directories only; the primitive is not a production lock.
Directory identity and integrity, filesystem semantics and durability are
assumptions; network filesystems and other hosts are not established.

Production cross-entrypoint deduplication remains blocked: existing entrypoints
do not acquire this marker and may use other output paths. A future reviewed,
authorized integration must make every permitted entrypoint share a fixed,
securely administered coordination directory and stable scope, account for
existing and historical attempts and noncooperating processes, seal the added
code and bind the final environment through launch. Lock failure or crash must
remain fail-closed; recovery or disposition of any existing claim requires an
explicit authority decision. GPU coordination, production authorization,
resource reservation, physical execution and saved-output acceptance remain
separate gates. No automated retry or claim cleanup is authorized here.
