# E3 integrated HOST and saved evidence candidate

This is not production-ready and does not provide full saved strict. It
integrates the existing authorized HOST launcher (real loader, source/model
guards and same-environment episode) with saved inspection and the partial
guard reconstruction. The original launcher clock, device, budgets, episode
and authorization remain unchanged. Nothing runs on import.

The authority's existing `additional_sources` must preseal the absolute paths
and SHA256 values of this module, the original launcher, the existing saved
consumer and the partial-guard consumer. Source hashing happens before any
HOST operation. The original control5/resolver pins and all original source
requirements remain in force. The shared authority is an input and is not
modified by this module.

`run` invokes the existing actual HOST path only after the exact original
authorization and source binding checks. It has not been run by this worker.
`inspect` rechecks the saved result; partials additionally go through partial
guard reconstruction. `require-strict` fails closed because the existing
consumer does not establish complete native snapshot and failure-path
semantics. A complete transport or matching recorded guard is not a full
strict or scientific admission. Transient module unloading remains outside
the two-point census evidence.

For authority-bound partial outputs, the integrated consumer also validates
each completed native observation's frame identity, ordering, original f64
body-snapshot bits and relation to native7. It preserves unmatched frames as
an explicit count, not as fabricated observations. If an episode-failure
event and completed terminal row both exist, their error payloads and order
must match. A failed or absent terminal row is not invented. This validates
recorded evidence, not the physical process or every partial failure path.
Fixtures cover a same-journal fake episode failure after observation and
reject altered snapshot and terminal error records.

For successful output, the integrated inspector additionally checks all
recorded observation snapshots and requires all frames matched and a durable
terminal row. It requires loader receipt, before guard, native records,
cleanup intent/return, after guard and launcher return in that order, and
rejects native events after cleanup begins. Metered close is deliberately
allowed within cleanup. These are necessary checks, not a full strict proof.

The real loader control-flow fixture invokes the original guard snapshot
function before and after the fake episode using the fixture model and fake
CUDA interface, and records the snapshots in the same journal. The same
loaded fake model is passed to the numerical backend and fixture encoding
marker; identity checks cover its fake parameter and device. The marker is
not the real encoder, and no actual HOST model, GPU state or production
episode is authenticated.

The integrated saved inspector reconstructs per-boundary encoder/projector
attempted and returned counts from event tokens and the physical encode
intent/return. It enforces the native hook prefix and causal boundary order;
partial pending work remains attempted and is not refunded. The old encoder
journal records only boundary and module role, not the object identity or
module-instance binding. Consequently matching guard snapshots and loader
receipts do not independently establish that the recorded encoder callbacks
came from the loaded model object. Full saved strict remains unavailable;
the same-object fake fixture is not a substitute for that missing evidence.

Fixture-only test from the repository root:

```sh
.venv/bin/python -m pytest -q tests/test_agent_e3_integrated_host_saved_20261029_v1.py tests/test_agent_e3_partial_guard_saved_20261029_v1.py tests/test_agent_e3_host_loader_episode_fixture_20261029_v1.py
```

The synthetic loader receipt is used only to exercise wiring; no real model,
CUDA, physics or upstream full strict is executed. Complete same-episode
guard/loader/cleanup fixture and full independent saved semantics remain work
before production acceptance. This candidate must be reviewed by MAIN.
