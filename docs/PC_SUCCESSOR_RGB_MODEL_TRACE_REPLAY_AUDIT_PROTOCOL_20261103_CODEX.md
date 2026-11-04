# Successor RGB model trace replay: development audit protocol

Replay the existing successor RGB development panel without changing its
selection, actions, outcomes or original records. For each of the 24 saved
decisions, reconstruct the model context from its recorded RGB frames,
action history, fixed checkpoint, normalizer, target, camera and calibration.
Feed every recorded `model_actions` batch through the same action-packet
unroll and record maximum absolute position and score differences. Verify
the original input and output receipt hashes, model parameter hash, action
normalization, prediction-token count, legal shapes, finite outputs and
policy source hash. Compare using an absolute tolerance of `1e-4`, reporting
actual maxima, failures and batch counts rather than silently filtering.

The replay checks current code and saved inputs against saved outputs; it
cannot retrospectively certify the hash of a dependency not bound at the
original execution. Record those current hashes explicitly as audit-time
observations, not original-run provenance. No physical outcome is used to
select or modify the tested batches. This is an engineering audit only,
does not create independent roots, and cannot establish PC1, PC2 or PC3.
