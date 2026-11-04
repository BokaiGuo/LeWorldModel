# P2 conditional suffix initialization: native CUDA development protocol

Before dispatch, seal a new plan with the existing verified B0 fixed context,
goal and all 16 candidates, the new conditional suffix bridge, frozen PushT
checkpoint and normalizer, explicit source hashes and meter implementation.
This uses one exposed development root. It is not the historical 40-candidate
SRA and is not an independent experiment.

For each same-bank candidate, compile a suffix from its predicted complete
intermediate context, preserving the initial prefix. Compare that initializer
to the unmodified joint-spectral candidate under the same eight-step
projected gradient optimizer and objective. Run all 16 candidates in order;
preserve attempts, failures and partial ledger and do not retry automatically.
Save all initializers, optimized actions, scores and accounting.

The composition-only initializer incurs extra prefix and suffix transitions,
16 Jacobians with 192 requested output directions each, and 16 bounded
solves. Meter model hooks and both optimizer callbacks by category and
report these additional costs. This is explicitly unequal total cost, not a
fair P2 advantage comparison; encoder, model loading, historical bank
generation, energy and FLOPs remain outside the measured scope. Freeze and
postcheck checkpoint parameters and bound source hashes.

This is model-only, not physical, not independent and not an admission gate.
Historical source drift in unrelated older SRA contracts is retained as a
limitation; do not relabel this control as that algorithm. An equal-resource
physical comparison requires a separate protocol.
