# Pinned AP CEM kernel: Reacher recorded-RGB CUDA smoke protocol

Before execution: use the same fixed archive inputs as
`pc_ap_reacher_cuda_recorded_smoke_20261102_v2`: Reacher memory episodes
0, 2 and 3, first 31 frames each; query episode 1 with observations at
steps 0, 5, 10 and goal at 35; action history uses recorded actions 0–9.
The query episode is explicitly excluded from memory. This is a development
smoke on exposed data; the archive, model-training membership and exclusion
of live roots from training are not certified.

Use the existing frozen Reacher checkpoint, original action stats, CUDA
encoding and predictor, unchanged observation-only memory retrieval,
five-action normalized bounds, a zero prior, seed 20261102, 16 candidates,
two CEM iterations and four elites. Use the new pinned-official-compatible
torch CEM kernel. The score callback bridges CUDA candidates through CPU
NumPy to the existing Reacher CUDA predictor; the bridge time is included
in planning elapsed time. Record predicted sample/batch counts, retrieval
identity, RGB hashes, bounds, action, costs, source and weight hashes and
before/after frozen-parameter hashes. Fail closed if a required input,
non-finite value, action bound or count check fails; never use CPU model
inference. The known archive ID check is not a provenance certificate.

This is one recorded decision, not physical execution, full official AP
replication, an equal-budget comparison, or evidence that P1 passed.
