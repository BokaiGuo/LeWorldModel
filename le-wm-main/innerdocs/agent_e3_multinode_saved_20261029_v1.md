# E3 multi-node numeric saved audit

Two fixed later nodes 10 and 15 share a single flat transport journal, with
explicit start/native/row/end boundaries. Independently replays complete HOST
arrays and 32-step numeric state per node; physical events can interleave but
numeric events cannot splice across nodes. Incomplete node, wrong boundary,
row chunk, changed descriptor or event order fails. Real temporary-disk tests
use fake CUDA algebra; no actual neural or physical operation.

This is a numerical consumer only: caller-supplied body/context is not itself
authenticated, and source/model/RGB, native lifecycle, actual encoder, cleanup,
publication and production CLI are not implemented. Not a full strict receipt or
scientific release. It does not consume the report/completion publication schema.
