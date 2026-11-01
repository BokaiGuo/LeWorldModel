# E3 transport PNG and publication

Fixture-only extension of unified transport: RGB224 PNG bytes/pixel hashes,
per-image 256 KiB and 32-image caps, and exclusive fsynced report/completion in
the 64 MiB reserve. Independent consumer rebuilds interleaved event chain,
referenced PNG/NPY inventory, category bytes, operation lifecycle, final hashes,
pending operations and inclusive publication boundary. An overrun marker fails
the all-and-only inventory. Tests use temporary files, no NN or physics.

This transport is not a full E3 saved strict: it does not reconstruct native
setup/actions, source authority, pixel preprocessing, encoder hidden/CLS,
numeric AD replay, model/precision/CUDA, or a same-env dispatcher. No production
CLI or scientific gate. Consumer rejection of failed completion is not a license
for a runtime to publish failed work as complete.
