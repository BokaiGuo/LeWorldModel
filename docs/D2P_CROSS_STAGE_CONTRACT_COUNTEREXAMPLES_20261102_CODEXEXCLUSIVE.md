# Planning Compiler cross-stage contract: counterexample audit

2026-11-02. CPU-only fixture audit; no gate changes, model calls, GPU, physical
execution, or historical replay. The companion test deliberately establishes
that the draft validator accepts malformed inputs; a passing test is evidence
of a defect, not evidence that the validator is sound.

## Findings

The draft is
`le-wm-main/experiments/spectral_reachability/d2p_p1_p2_p3_interface_contract_20261102_codexexclusive.py`.
Its accepted receipts have `P1_admission`, `P2_admission`, and `P3_admission`
set to false, but the following additional assurances are not enforced:

| Priority | Reproducible behavior | Consequence and specific next patch |
|---|---|---|
| P1 | Three evidence fields are naked booleans; a bound outcome row may have `status=failed`. | Require per-component references with matching root/task/model/protocol and source artifact hashes; validate the bound outcome status and prevent failed/partial evidence being called complete. Booleans and hash syntax alone cannot verify an artifact. |
| P1 | `source_disjoint_outcome` and `outcome_sealed_before_selection` are caller assertions. | Join independently stored pre-outcome manifest and source identities; prove set disjointness over relevant source lineage, not just a flag. A timestamp or hash by itself does not prove when sealing happened. |
| P2 | Multiple edges can have disconnected observed/output-to-next-input contexts. | Define canonical context format and digest; verify each continuing edge's actual observed context binds exactly to the next edge's compiler input, after any explicitly recorded conversion. Do not compare predicted postconditions to observations as a proxy for identity. |
| P2 | Required edge fields may all be `None`; failed edges can assert successful recompilation. | Validate nonempty context/witness/replay-receipt structure and status-dependent event sequences; retain partial/failed attempts in the census without demanding a recompile when execution did not reach observation. |
| P2 | `search_executed`, equal budget/feedback and comparator presence are flags. | Bind actual search, comparator, budget and feedback receipts to frozen candidate/branch census; use the existing P2 matched-composition and resource-ledger work instead of inventing another accounting implementation. |
| P3 | Negative/fractional horizon, depth and budget are accepted; signal budget need not match planned allocation. | Require positive integer horizons/depth, integer budget in declared units, and reconcile rule output with planned allocation and actual attempts. Explicitly distinguish cap, allocation, attempts, failures, and unused capacity. |
| P3 | Pre-outcome timing and no-refund are caller assertions. | Bind a predeclared rule/config hash and ordered signal/decision/attempt/outcome events; reconcile all attempts with instrumentation. Event ordering within a caller-supplied log is not independent proof of sealing or completeness. |
| Cross-stage | `contract_rejected` may be returned normally by `validate`. | Reject this status for a positive validation API, or return an explicit non-accepted result; do not infer acceptance merely from lack of exception. |

The earlier gap register describes a *proposed interface*, not the full
implemented coverage. In particular, it should not be read as claiming
source-disjointness, physical evidence, observation authenticity, or temporal
sealing are verified. Existing `d2p_composition_witness_v2` performs exact
continuity checks in its own **model replay** contract; this does not
automatically validate this draft's actual-observation fields.

## Reproduction

From `le-wm-main`:

```
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider \
  tests/test_d2p_cross_stage_interface_counterexamples_20261102_codexexclusive.py
```

The companion test uses fabricated in-memory inputs only, with intentionally
invalid values. The expected current outcome is that all tests pass because
the defective validator accepts each counterexample. Once an independently
reviewed fix is made, these negative audit assertions must be inverted or
replaced by rejection tests; no scientific gate is affected.

## Scientific and implementation boundary

P1 still needs eligible tasks, same-root non-greedy/own-state/real outcome
evidence, disjoint confirmation and strong controls. P2 still needs real
multi-program search and matched comparator results. P3 still needs real
signal-driven scheduling and a full matched success/cost comparison after the
required preceding stages. This audit proves neither a stage result nor that
all other implementation defects have been found.
