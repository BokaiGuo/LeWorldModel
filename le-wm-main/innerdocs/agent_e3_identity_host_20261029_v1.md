# E3 identity-bound HOST candidate

This new, independently named candidate wires `capacity.load_bound` into the
existing real HOST loader path, then sends its held model object to both the
same-environment numerical backend and `Binding.encode`. The episode preserves
the original setup, reset instrumentation, guard snapshots, native meter,
cleanup and terminal-row transport. It neither modifies nor impersonates an
old output schema as a new successful receipt. A new identity-host marker is
required by `inspect`, so old journals fail closed.

Inputs are the original explicit absolute authority path and SHA, with the
new host, capacity and identity source files added to the authority's source pins in
addition to all earlier integration pins. The existing original HOST
authorization is required by `run`. `inspect` consumes the existing saved
evidence and the new identity contract, and checks successful physical encode
boundaries against identity returns. It always reports `full_saved_strict`
and scientific admission false; partial before binding is preserved as
partial, not a successful identity receipt. Calls failing before the host
marker (e.g. authority read failure) cannot satisfy this new inspector.

Budget: at most one bind record and two encoder boundaries with at most four
identity records each: 9 records, each bounded by the unchanged 16,385-byte
event cap, or 147,465 bytes maximum. Before model loading, the host performs
the existing physical-events and aggregate cap check for that amount. This
is a predispatch check, not a separate reservation and not a proof that the
whole episode will fit: original 512 MiB total, 64 MiB reserve, 32 MiB
physical-event cap, 200 MiB floor and 1800-second clock are unchanged.
Additional hooks and journal fsync work consume the same wall budget. The
real loaded model's full parameter-instance binding record has not been
measured and may exceed the existing per-event cap. The new capacity path
measures the loader-returned object before binding, records the result after
binding and saved inspection independently compares the journal line size.
If it does not fit, the path fails before creating an environment; no cap or
field is silently changed. A production-ready size proof requires the actual model inventory
and the complete original event/metadata budget under MAIN's authorization.

Fixture-only tests from the repository root:

```sh
.venv/bin/python -m pytest -q tests/test_agent_e3_identity_host_20261029_v1.py tests/test_agent_e3_identity_capacity_20261029_v1.py tests/test_agent_e3_encoder_identity_20261029_v1.py
```

Tests cover source-pinned authority, pre-loader failure and old-log rejection,
setup exception/cleanup/terminal persistence, loader/identity fixture, and
saved corruption rejection. They do not exercise the new full real encoder,
two-node same-environment HOST lifecycle end to end. No neural or physical
production, actual model load, or GPU allocation was performed. This candidate
must not be treated as ready for MAIN production until those fixtures and the
actual shape/size proof are reviewed. The original full saved strict gap and
source-trusted, non-cryptographic identity limitation remain.
