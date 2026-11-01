# E3 encoder instance identity contract candidate

This is a new forward-only contract. It cannot add identity evidence to old
logs and is not integrated into the existing frozen launcher. No production
run is authorized by this file; MAIN must review and explicitly integrate it.

`load_bound(prepared, torch, device, store, prepare_sha)` calls the original
HOST loader with its real journal bridge and immediately retains the returned
model object, encoder/projector objects and named frozen parameter object and
device identities. The preparation and source closure must already have been
authenticated by the caller. `Binding.encode(torch, frame, boundary, check)`
uses that same held model and calls the original encoder primitive; it checks
module/parameter identities at entry, in pre-hooks and after return. Distinct
encoder/projector callback records are journaled, then the wrapper removes
its hooks and records return or bounded failure. No original globals are
patched. `_encode` is an explicit fixture seam, not the production method.

`consume(output, report_sha, completion_sha, prepared, prepare_sha,
source_bindings)` validates the publication, rehashes the named source closure,
checks the original loader receipt, and reconstructs binding and per-boundary
callbacks, matching them to native encoder intents on successful returns.
The caller must independently authenticate `prepared`, its SHA, and every
source binding; passing arbitrary mappings is not an authority. Partial
pending/failure is retained, not certified as success. Older logs lack the
required binding and are rejected.

Object addresses and the random per-binding tag are runtime correlation
evidence produced by source-trusted code, not cryptographic proof of process
memory or protection against a compromised producer. This is not neural or
physical replay, class-closure proof, or full saved strict. Original resource
caps, failure reserve, scientific gates and 1800-second clock remain the
integration caller's responsibility. Pre-register extra identity event bytes
and callback work before any production use; this module does not establish
that existing event budgets accommodate them.

Fixture-only regression from the repository root:

```sh
.venv/bin/python -m pytest -q tests/test_agent_e3_encoder_identity_20261029_v1.py tests/test_agent_e3_launcher_loader_bridge_20261029_v1.py
```

Fixtures call the real loader control flow on invented tiny files with a fake
model, deserializer and device, and use the explicit implementation seam; no
real encoder, CUDA allocation, physical environment or production is run.
They test swapped module/parameter objects, callback failure and hook cleanup,
plus coherent event-level identity, loader, order and device mutations. They
are not a complete actual encoder/HOST end-to-end fixture.
