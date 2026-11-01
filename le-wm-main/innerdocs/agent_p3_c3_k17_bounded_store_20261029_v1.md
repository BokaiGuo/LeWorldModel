# Prospective K17 engineering storage contract

This independent candidate proposes 128 MiB output, 8 MiB reserved failure
space and a 200 MiB free-space floor. It does not alter the older 64 MiB
contract, set a scientific budget, or claim equal compute. It is not a runner.

At most 306 PNG files of at most 151,834 bytes each; journal at most 32 MiB;
execution binding at most 2 MiB; report at most 16 MiB; completion at most
2 MiB; separate failure record at most 8 MiB. Their integer sum is below
128 MiB. The quotas cover on-disk copies, including any duplication deliberately
placed in these files. The bounds do not establish that arbitrary native
snapshots can be represented within the quotas: an oversized item fails before
its write, and a later adapter must preserve all six statuses as far as its
failure reserve permits. Actual I/O failure can prevent even failure recording.

This implements the storage primitive, not the native adapter, wall meter,
PNG decoding, full event/source/goal/action audit, or a production consumer.
No physical dispatch is authorized by this document. A subsequent adapter
must validate its actual serializations and explicitly report a quota failure;
no evidence may be silently discarded to fit.
