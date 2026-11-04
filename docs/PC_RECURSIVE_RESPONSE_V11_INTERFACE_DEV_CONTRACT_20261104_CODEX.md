# Recursive response v11: development interface contract

This is a new development candidate, not a frozen confirmation method or a
claim that any PC gate has passed. Previously stopped panels remain stopped.

The physical action step, model packet, search segment and dispatched chunk
are separate units. A predictor accepts only whole packets. Each search
segment and the remaining fixed-terminal horizon must be a positive multiple
of the declared model packet; unsupported combinations fail before prediction.
No action is truncated or padded. The prediction-token ledger counts actual
batch size times physical action length divided by packet length, including
baseline, carry, probes, candidates, refinement and paired abstention. The
caller independently meters the predictor and must reconcile both ledgers.
Encoding, GPU time, differentiation, SVD and orchestration costs remain
separate resource axes, not included in prediction-token equivalence.

At the first decision carry is absent. Subsequently the caller supplies the
unexecuted tail, exactly the remaining horizon, in the same bounded physical
action units as newly generated actions. Zero and carry are re-evaluated in the
new observation before any search. Every evaluated full chain can become the
incumbent; full-chain refinement starts at that incumbent and retains it.
After refinement, the selected incumbent and its first-segment-zero
counterfactual are jointly rescored; the lower score is dispatched. This
paired abstention is not evidence of a physical mechanism, which must be
measured separately on the final actual chain.

Initial verification uses deterministic synthetic predictors only to test
contract behavior. It does not reopen the stopped E0 or JEPA panels, establish
task feasibility, or support a performance or PC0–PC5 claim. A real feedback
worker and independent actual-call reconciliation must be integrated and
audited before claiming an end-to-end v11 path.
