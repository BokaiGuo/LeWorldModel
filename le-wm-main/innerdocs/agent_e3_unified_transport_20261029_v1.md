# E3 unified tensor and operation transport

Single bounded uncompressed NPY store and one hash-chained JSONL for interleaved
physical, numerical and encoding operations. Numerical arrays retain the original
768-element cap; encoder arrays explicitly accept original RGB224, preprocessed
frame, full hidden `(1,257,192)`, projector IO and native192. No arrays deduplicated.
Each group has a separate write-before cap; total working data retains 64 MiB of
512 MiB and 200 MiB disk floor, and write checks use a common 1800 s clock.
Consumer bounds the reads, checks chain/attempt/return/failure, all and only
referenced array files, SHA, NPY shape/dtype/finite/EOF, and independent group
usage. Failures and pending attempts cannot be complete. Tests use temporary files.

This is a transport consumer only, not the requested whole-experiment strict:
it does not validate PNG/physical task, native event semantics, preprocessing,
encoder/projector connection, numerical replay, source/model, report/completion,
or live CUDA. It has no physical PNG writer, production dispatcher or CLI and is
not authorized for production. Final reserve publication and combined lifecycle
must be implemented and audited before use. No scientific admission.
