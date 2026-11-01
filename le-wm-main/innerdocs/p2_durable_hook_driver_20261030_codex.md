# P2 durable hook driver: CPU-verified interface

The opt-in `Driver` connects the existing `HookBridge` to the existing
`agent_p3_c3_k17_physical_transport_20261029_v1.Journal`. The latter creates an
exclusive file, chains canonical events, flushes and `fsync`s each event, and
fsyncs the containing directory at creation. The new driver creates an exclusive
absolute directory and records contract, identity, source hashes and projected
caps in its first event. It neither imports nor invokes a model or simulator.
It accepts caller-provided callbacks through `call(scope, resource, fn, ...)`.

The native public ledger sends an attempt to the durable sink **before**
invoking `fn`; a full projected cap instead records `resource_blocked` before
raising. Failed or blocked calls make this driver fail-stop. Existing ledger
attempted counts are nonrefundable; terminal events are recorded on return or
failure. `finish()` emits a final ledger receipt, closes the journal, and
returns its SHA256 for independent retention. A journal write or fsync
failure can leave a partial file: a failed journal cannot be sealed, and a
missing terminal is rejected. No automatic retry, truncation, overwrite or
resume occurs. The byte limit does not reserve terminal capacity, so an
otherwise complete job may end with a rejected partial journal if it runs out
of space.

The read-only `verify(path, expected_journal_sha, expected_contract_sha)` checks
the actual hashed input bytes, canonical chain and order, external contract
binding, current direct source hashes, projected caps, independent ledger
recount, attempt/terminal bijection and nested event ordering. It reports
`failed_or_blocked` for terminal failures. The digest must be independently
retained; if an adversary can replace both the file and its expected digest,
the hash chain is not authentication. The contract digest also needs
independent prior retention; hashing alone proves neither preregistration time
nor source execution. Source checks cover the four direct imported modules and
this driver, not every transitive dependency.

Only the five bridge axes are mapped to P2 caps. Other caps, inherited work,
calls bypassing the bridge, physical work, FLOPs, end-to-end wall and peak
device usage are not guarded or certified. Hashes and fsync establish neither
coverage nor truthfulness of callback metadata or absence of hidden work.
The driver is an interface for a separately implemented new caller; sealed H15
and CEM drivers enforce an exact ledger type and cannot accept it unchanged.
There is no P1/P2 admission, full-vector predispatch guard, equal-compute or
fairness conclusion.

CPU verification:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_p2_durable_hook_driver_20261030_codex.py \
  tests/test_p2_hook_contract_bridge_20261030_codex.py \
  tests/test_p2_preregistered_resource_vector_20261030_codex.py
```

An externally retained journal and contract can be verified with:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m experiments.spectral_reachability.p2_durable_hook_driver_20261030_codex \
  --journal /absolute/path/events.jsonl --sha256 JOURNAL_SHA256 \
  --contract-sha256 PRECOMMITTED_CONTRACT_SHA256
```

The tests use CPU callbacks and temporary directories, not a real P2 execution.
