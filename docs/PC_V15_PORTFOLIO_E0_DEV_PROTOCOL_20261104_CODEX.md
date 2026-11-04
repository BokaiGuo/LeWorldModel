# v15 response/global-search portfolio: E0 development protocol

Status: prospective development diagnostic, not a confirmation protocol.
The source v13/v14 panels remain stopped. The hypothesis is that reserving
some full-action optimization for the same unprivileged global bank may
avoid some search-allocation losses seen in the exposed v13/CEM comparison.
This is not a claim that response structure, reconfiguration, or v15 improves
performance.

## Fixed design

Use exact Wall dynamics and four deterministic tasks: start `x=-0.5`,
`y` in `{-0.21, 0.21}`, heading in `{-0.9, 0.9}`, goal `(0.5, 0.21)`.
Check exact task tuples against the prior panels used by the existing
new-feasible-Wall runner. Historical disjointness beyond those panels and
training provenance are unknown. Failure of that check stops the run.

Three paired arms share one initial global proposal bank and each has a
hard cap of 2,409 five-step prediction tokens, horizon 15 and legal
normalized action box `[-1,1]`. Seeds are deterministic by task and arm.

* `response`: unchanged v13, cap 2,409.
* `global_cem`: the shared proposal bank followed by full per-step CEM and
  the same paired zero-prefix comparison used by the existing control.
* `portfolio`: unchanged v13 cap 1,920, followed by 160 full-action CEM
  proposals (480 tokens) initialized from the best shared global proposal.
  Re-score v13's selected chain, CEM's incumbent and its zero-prefix variant
  together (9 tokens), dispatch the lowest-scoring complete chain.

Record every actual prediction batch, action, endpoint and score. Report
terminal distance, strict success `<0.05`, selected source and actual token
counts per arm and task. Check the selected chain was actually scored and
is a best scored full-chain incumbent for deterministic exact dynamics.
Count all generation, optimization and final comparisons. This is a
prediction-token comparison, not a matched total-cost comparison: runtime,
SVD and other work are recorded separately where available.

## Stop and interpretation

Four roots and three arms are the entire panel; failures count as failures.
No replacement, retry, adaptive budget or expansion on this panel. Check
saved endpoint/score traces with the prior alternate geometry, which is a
same-environment implementation consistency check, not independent
replication. Seal source and protocol dependencies. The panel is development
data, not independent confirmation, and does not test final-chain PC2
interventions. All results, including ties and failure, must be retained.
