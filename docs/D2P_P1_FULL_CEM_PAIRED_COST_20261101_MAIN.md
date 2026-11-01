# P1 full-controller paired development costs

The four sealed reports were re-read by
`experiments/spectral_reachability/p1_full_cem_cost_compare_20261101_main.py`.
It verifies report hashes, each pair's root/goal binding and unchanged model,
the chronology of sealed nodes, and structured-node forward hook counts against
the dispatch ledger. It does not rerun models, reauthenticate every source,
measure GPU kernels or certify equal resources. Five synthetic contract tests
pass, including root, model, hook and chronology rejection cases.

| Development root | Controller | Native success step | RGB encodes | Predict/action encode | Backward/VJP | Jacobian/SVD |
|---|---|---:|---:|---:|---:|---:|
| 0 | structured | 15 | 3 | 549 | 720 | 3 |
| 0 | local CEM | 15 | 3 | 390 | 0 | 0 |
| 1 | structured | 14 | 3 | 549 | 720 | 3 |
| 1 | local CEM | 6 | 2 | 325 | 0 | 0 |

Counts are sealed dispatch upper bounds, not full cost, and reflect early
termination. CEM's replay and selection events are **not** charged on the
same ledger axes as structured's events; its zero ledger values do not mean
no selection or candidate evaluation occurred. Model loading, construction
and shared upstream work are not included in the node counts. The CEM report
also inherited inaccurate generic caps/hook descriptions, corrected only in
the separately prepared v2 contract, not retroactively in these results.

These are previously observed goals and roots, not independent confirmation;
no budget curve or equal-budget advantage can be inferred. The observed
full-CEM outcomes make a strong baseline necessary, including on independent
tasks, instead of relying on the favorable conditional-suffix comparison.

## Comparator chronology guard

The saved-report consumer formerly skipped `not_started` entries before
checking executed boundaries. A malformed roster with a sealed node after an
unstarted node could therefore pass if its remaining sealed boundaries matched
the expected sequence. It now rejects that ordering, and a regression test
exercises the failure. Six comparator tests pass; re-reading the four
hash-bound existing reports succeeds with unchanged reported numbers. This is
consumer hardening, not evidence that the historical reports had the defect,
nor a new neural or physical experiment.
