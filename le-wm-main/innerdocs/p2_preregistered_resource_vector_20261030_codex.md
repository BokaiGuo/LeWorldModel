# P2 prospective resource-vector comparison (CPU-only)

Existing `p2_full_resource_matching_20261027_v1` provides resource arithmetic,
nonrefundable reservation and a measured-vector comparison; its supported
guard rejects finite caps on axes the backend cannot enforce.
`agent_downstream_resource_contract_20261029_v1` maps selected saved counters
to a known/unknown ABI. Neither establishes this new full-coverage,
common-cap, identity-bound prospective comparison. This module is an opt-in
receipt validator, not a runtime guard, recorder, or scientific admission.
Existing producers and sealed files remain unchanged.

## Contract and receipts

The caller creates a canonical JSON contract (`sort_keys=True`, separators
`,` and `:`, UTF-8, no trailing newline, no NaN), independently commits its
SHA256 **before** any compared execution, and obtains separately SHA-bound
receipts. Hashes bind bytes but cannot establish commitment time. The contract
has exact fields `schema`, `arms`, `scope`, `identity`, and `axes`.
`schema` is the module name plus `.contract.v1`; scope is
`per_root_per_arm_standalone_planning_and_selected_physical_execution`.
Identity comprises root, task, model, action-domain, hardware and meter SHA256
fields named in `IDENTITY`. An arm's standalone charge must include its
attributable inherited setup; shared-work attribution must be documented in the
external evidence and must not be summed across standalone arms.

All axes in `AXES` are compulsory. A contract axis contains exact `unit`,
`aggregation` and finite nonnegative integer `cap`. The peak-device axis uses
`max`; the others use `sum`. Wall means whole planning elapsed time per arm,
including CPU work, synchronization and audit overhead; it must not be formed
by summing overlapping nested timings. FLOPs are measured/instrumented
floating-point operations under the common meter, not inferred from VJP counts.
The vector intentionally carries overlapping quantities separately, never as a
summed total. It does not cover every conceivable compute or physical cost.

Each receipt has exact fields `schema`, `contract_sha256`, `arm`, `run_id`,
`identity`, `status`, `axes`. Its schema ends `.receipt.v1`; it must be
complete, bound to the exact contract and identity, and unique per arm and
run. Each count axis has `unit`, `coverage`, `evidence_sha256`, `attempted`,
`completed`; each gauge substitutes `value` for the last two fields.
Coverage must be `complete`; attempts equal completions, every value is a
nonnegative integer within the common cap, and evidence is a SHA256 reference.
The references are declarations: this validator does **not** open the
evidence, establish hook coverage, verify chronological commitment, or
independently reconstruct runtime journals. Such verification and a
pre-dispatch, nonrefundable enforcement backend remain required.

`compare` returns separate `common_cap_satisfied` and `actual_vector_equal`
flags. Both refer solely to accepted receipt assertions and this fixed vector;
they do not prove equal FLOPs beyond the declared measured axis, equal total
compute, same action/state, causal effect, or P1/P2 admission. Missing,
unsupported, failed or over-cap data raises an error rather than becoming zero.
No dummy padding is authorized.

Example CLI (paths and hashes are placeholders, not actual preregistration):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m \
  experiments.spectral_reachability.p2_preregistered_resource_vector_20261030_codex compare \
  --contract /absolute/contract.json --contract-sha256 CONTRACT_SHA256 \
  --receipt /absolute/arm_a.json ARM_A_SHA256 \
  --receipt /absolute/arm_b.json ARM_B_SHA256
```

`historical-diagnostic` reads the pinned prior saved report only; it reports
missing or incompletely instrumented axes and refuses to retrofit a
prospective contract. It does not revalidate the entire upstream source chain.
The preceding saved report itself was produced by rebuilding its consumers.

Tests include synthetic positive and negative contract cases, and the real
historical report is checked separately. Synthetic validation is not an actual
resource measurement or experiment.
