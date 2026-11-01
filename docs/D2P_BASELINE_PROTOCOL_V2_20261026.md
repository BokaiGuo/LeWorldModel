# D2P strong-baseline matched-cost protocol v2 — 2026-10-26

## Outcome and boundary

Implemented an independent metadata-only matched-cost sidecar and serial reservation ledger. This closes one bounded readiness gap: baseline identity, common protocol bindings, planned resource ceilings, measured resource vectors, feedback records and failed attempts can now be checked together without editing sealed kernels or runners. It does **not** implement GRASP, GN-MPC or kinodynamic search, independently attest CUDA execution, or establish physical efficacy. P1 remains NOT PASSED. No dependent scientific stage is opened.

New implementation: `le-wm-main/experiments/spectral_reachability/d2p_baseline_protocol_v2.py`.
Tests: `le-wm-main/tests/test_d2p_baseline_protocol_v2.py`.
No existing source was edited by this task. No training, adapters, external downloads, GPU launches, commit, push or deletion.

## Audited evidence and selected gap

Read the requested proposal, `D2P_BASELINE_FAITHFULNESS_AUDIT_20261026.md`, `D2P_BASELINE_EVIDENCE_POLICY_20261026.md`, `D2P_BASELINE_PROTOCOL_FIELDS_20261026.md`, `D2P_SAMPLING_BASELINE_CONTRACT_REVIEW_20261026.md`, and the GRASP/Hi-LeWM source audit. Inspected current `d2p_sampling_baselines.py`, `d2p_cuda_sampling_cost.py`, `d2p_model_call_instrumentation.py`, and P0 action-GD/restart-GD/CEM implementation paths.

- Current restart-GD optimizes action parameters with Adam; it is not evidence of official GRASP's algorithm. The local source audit has not established official-paper/version/mechanism identity. Do not infer publication facts or performance from the proposal.
- P0 GN/D2P common-solver equivalence is an implementation control, not an independent algorithm victory. Kinodynamic comparison needs explicit edge dynamics, expansion/selection rules and stopping limits; this task does not pretend to supply them.
- Current sampling kernel explicitly labels clipped-Gaussian MPPI a heuristic. It now contains source-dtype checks and attempted/failed counters: the older review's dtype finding must not be asserted as an unfixed current defect. This task does not change those concurrently maintained files.
- Candidate-cost counts and predict samples are different units; neither alone measures backward, encoding or wall cost. Existing predict instrumentation explicitly excludes encoder/backward work and cached/compiled bypass paths.

Therefore a shared resource/protocol contract is a useful prerequisite to future faithful comparisons, without making an unsupported official reproduction.

## API and strict contract

`protocol(...)` requires all SHA256 bindings: root, checkpoint, source, observation access, goal/metric, action domain, temporal execution, evaluation, calibration/outcome manifests, cache policy, hardware/concurrency and measurement scope. Unknown values cannot match each other. Calibration/outcome manifest hashes must differ, but actual root disjointness requires an external manifest audit: different hashes alone do not prove disjoint sets.

Action-domain artifacts must distinguish parameter box from effective normalized/physical action support, mapping (including tanh), dtype, initialization and proposal law. Temporal artifacts must include horizon units, primitive-step conversion, action repeats, chunks, execution limit and stopping criteria. Binding hashes are syntax-checked, not file existence/content-verified. Independent audit must establish the claimed contents before real use. A single common `source` binding should identify the whole frozen comparison bundle; per-method implementation hashes are separate.

Explicit feedback steps are positive increasing primitive-step indices; `[]` means open loop. All resource vectors must contain every coordinate, including measured zeros. Unknown work must block integration, never be encoded as zero:

`predict_calls`, `predict_samples`, `predict_token_work`, `encoder_samples`, `scalar_backward_calls`, `jacobian_vjp_rows`, `jvp_calls`, `candidate_cost_samples`, `search_expansions`, `planning_wall_ns`.

The measurement-scope artifact must define logical backward/VJP/JVP units, token-work formula, warmup/cache amortization, compilation, rejected/failed proposals and synchronized timing boundaries. No conversion to FLOPs is implemented. Wall-time caps require conservative reservations and CUDA synchronization in the external runner; a metadata ledger cannot preempt an overlong kernel.

`compare_protocols(left, right)` requires exact shared bindings, feedback schedule, primary-resource names and primary ceilings. It discloses nonprimary ceiling differences. Equal ceilings do **not** imply equal actual use, equal wall cost, or fairness under every resource dimension. No paired inference, pooled results or scientific admission is granted.

`method_identity(...)` distinguishes `proxy`, `local_algorithm`, and `faithfulness_audited`. The last label requires a hash for an external audit mapping pinned official source, essential mechanisms, hyperparameters and deviations. It is a caller claim; the module never authenticates it and always returns `official_reproduction_verified_here=False`. Existing restart-Adam should remain `proxy` with an unmistakable name such as `restart_adam_not_GRASP`. Do not attach paper-reported scores to this identity. Published values remain in the separate reported-baseline evidence registry; this module accepts no performance values.

## External runner integration contract (not launched or wired here)

1. Freeze and independently validate the protocol artifacts, calibration-only choices, outcome manifests and method audit. Coordinate any host-CUDA launch with the parent. Use one ledger per root/method; no CPU or remote-CUDA fallback. `execution_backend='host_cuda'` is required metadata, **not** proof of actual device placement.
2. Call `reserve(purpose, full_upper_bound_vector)` before dispatch; charge all proposal screening, initial/nominal rescoring, line search, Jacobians, rejected search edges, compilation and cache building. Ledger disallows overlapping reservations; concurrent runners need isolated ledgers and externally matched contention policy.
3. Execute with independent instrumentation, then `complete(ticket, actual_vector)`. Exceeding a reservation poisons the ledger even below total caps. Exceptions use `complete(ticket, succeeded=False)`; reserved attempted work remains charged, and no best-ever fallback can convert the failed attempt into a completed receipt. External callbacks must report exceptions explicitly; the ledger does not intercept model calls.
4. Record `feedback(step)` only after actual scheduled RGB feedback has been obtained. The ledger checks ordered reported timestamps and no pending work, not observation authenticity or the placement of all work around feedback. Each event's purpose/external trace must establish causal stage alignment; timing records alone do not prove access parity.
5. Finalize once with `receipt(finalize=True)`. Empty ledgers, pending work, failed/over-budget attempts and incomplete feedback cannot complete. Finalization seals further dispatch; nonfinal receipts never claim completion. Receipts carry protocol fingerprint, detached event history, identity and charged resource vector. `completed=True` means this metadata accounting contract only, not a trustworthy physical result.

Keep evaluator physics/metrics in a separate trace; do not bill evaluator-only state access as planning privilege or silently exclude true planning feedback encoding. Parent retains the official-vs-object metric audit. This module intentionally neither reads physical outcomes nor changes any P1 gate.

## Validation

Command from `le-wm-main`:

```sh
.venv/bin/python -m pytest -q tests/test_d2p_baseline_protocol_v2.py
```

Initial result: **38 passed in 0.09s**. Fixtures cover every binding mismatch, unknown metadata, split aliases, invalid resource counts, feedback mismatch, primary/nonprimary budget distinctions, claim labels, host-CUDA metadata requirement, copy isolation, full-vector accounting, pre-dispatch caps, failed attempts, reservation overruns, invalid measurements, pending work, missing feedback and ticket replay. These are standard-library metadata fixtures, not neural CPU fallback, physical experiments, or baseline efficacy evidence.

Combined regression command adds `tests/test_d2p_sampling_baselines.py`, `tests/test_d2p_model_call_instrumentation.py` and `tests/test_d2p_reported_baselines.py` to the above invocation: **217 passed in 0.37s**. These existing tests were executed without editing their files; passing them does not certify external instrumentation coverage or official-method faithfulness.

Remaining readiness gaps are explicit: no faithful official GRASP audit, no independently wired strong GN-MPC/kinodynamic arm, no complete external backward/encoder/timing instrumentation, and no newly executed held-out physical comparison. This sidecar is intentionally opt-in; old sealed sources, historical receipts and published-evidence registry semantics are unchanged.
