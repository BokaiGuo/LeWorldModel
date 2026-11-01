# Metered spectral8 H3 initializer and raw suffix adapter

This module is a callable adapter, not a production launcher and not yet
connected to the existing four-arm host entry. It does not load checkpoints,
persist events itself, execute physics, prove fairness, or grant P1 admission.
The existing ridge proxy remains separately labeled and unchanged.

For an already constructed structured `LeafAdapter`, use
`policy(mean_list, std_list, history_size=3)`, then create the prospective
four-arm contract and durable driver before any calls. Set the structured
initializer hash to `file_sha(module.__file__)`, the common optimizer hash to
`file_sha(codex_four_arm_raw_suffix_20261030_main.__file__)`, the common
optimizer configuration hash to `spec.digest(config)`, and
`phase_policy_sha256` to `spec.digest(policy)`. The caller must bind and check
the real root, task, checkpoint, normalizer, context, source closure, precision,
and hardware; the adapter only checks the indicated contract digests, arm,
dimensions, float32, positive stats, and identity projection. Existing
`spec.preflight` validates the contract structure, not those input claims.

`required_caps(config)` returns the structured arm's complete-run reservation
for the explicit axes. The driver must have at least that amount remaining
before entry. Invoke
`run(leaf, states, normalized_history, goal, mean, std, projection, config, policy, contract, emit=durable_sink)`.
The sink must durably persist each event before return and raise on failure:
signal, full raw spectral bank, all post-update candidates, final scores and
selection. Caller must finalize the driver even on error and must not report a
partial path as success. This callback interface cannot itself attest that the
sink is durable or separately consume its output.

The rule matches existing spectral8 H3 algebra: nominal raw-zero 192x30
Jacobian; identity Q; float64 SVD; signed first-argmax convention; radii
sqrt(5)*(.25,.75), two leading modes and signs -1,+1; float64 tanh then one
float32 cast. Leading near-degeneracy is additionally rejected at relative
1e-10; this refusal is a new adapter gate, not a change to the old bank.
This is an H3 spectral8 engineering extension, not a claim of equivalence to
original R or of an independently authenticated D2P result.

For S suffix steps, each completed structured run charges exactly
`3 + 24*S + 24` action encoder calls, predict calls and predict samples;
`192 + 8*S` backward calls and VJP rows; one Jacobian marker, one actual SVD,
eight final replay markers and eight selection evaluations. All are
non-nested dispatches; model and AD failures are not refunded. Q@J is inside
the paid SVD callback, but matrix arithmetic, size, transfer, serialization,
wall-clock, energy, upstream context encoding and internal model operations
are not separately metered. The spectral bank, suffix and replay are not
evidence that the remaining three arms have equal total costs. No real model
or GPU execution was performed for this adapter.
