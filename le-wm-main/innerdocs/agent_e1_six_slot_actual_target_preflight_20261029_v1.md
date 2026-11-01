# E1 v2 real-source preflight and fixed-shape risk audit

This is saved-only/fixture validation; it is not a CUDA run, physical run,
complete production receipt, or scientific admission. Frozen v2 was not edited.

The actual six-target prepare is rebuilt by `v2.authorities()` from its fixed
external SHA `cd96f6e28b25d3b7e3b6649c760b2f111747aec5c0ce9c8a41741318209df8a0`.
The six roots/arms, task/blind bindings and each 768-byte finite little-endian
float32 target are checked. The learned targets originate from the saved Hi
FIRST NPY: upstream `join` checks original file and data SHA, FIRST-to-low
byte equality, receipt fields and root identity; no historical Hi scaler or
H10 original learned training claim follows. The new fixture uses each actual
learned target in the full 72-candidate controller with an artificial task and
artificial non-neural backend. It verifies all 144 operation tickets and saved
math, result-file roundtrip and swapped-target rejection. It is not actual
neural inference or a physical counterfactual.

CLI prepare/verify were executed exclusively in `/tmp/e1-v2-preflight.4ahSUk`:
the prepare SHA was `df548a6a9d8356f654e6aac823e7dd90383570da56631c38155b539bad8ae0dc`;
verify succeeded; an all-zero expected SHA was rejected. This is temporary
preflight, not a production prepare authority.

The prior saved native four-arm journal `audit-io` checks hashes and all
6,085 records, with max metadata 1,680 B and max unique f64 array bytes 8,000 B
per event. The fixed config sets AE/predictor/projector widths 192 and context
3, but native runtime hook checks only some shape dimensions. The 16 KiB/event
metadata, 32 KiB/event new arrays, 1 MiB/payload arrays, 4 MiB/payload metadata,
10,000 events, 27 payloads, 1 GiB aggregate and 200 MiB free floor are runtime
guards, not an a priori proof that new model-return values always fit. Array
dedup never omits module IO; reader checks reference hashes and all stored bytes.
On oversized event, archive append may precede the guard and leave unreferenced
bytes; the run fails, strict consumer refuses that partial, and no retry occurs.
The separately reserved 64 MiB supports normal failure/report publication but
cannot guarantee durable reporting on arbitrary disk or fsync failure. Peak CUDA
allocator limits are sampled/checks, not hardware-enforced total-device limits.

The timing witness verifies completion file and directory fsync were completed
before its timestamp, but not its own subsequent fsync. No production run is
authorized by this document. For MAIN CLI use the frozen v2 doc.
