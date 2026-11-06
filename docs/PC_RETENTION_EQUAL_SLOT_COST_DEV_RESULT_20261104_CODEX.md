# Equal-slot retention and full-cost development result

Date: 2026-11-04

This report covers the new 16-root development task family specified in
`PC_RETENTION_EQUAL_SLOT_COST_DEV_PROTOCOL_20261104_CODEX.md`. It is not an
independent confirmation and does not establish PC2 or PC3.

## Audit

- 16 paired roots × 5 arms = 80 completed plans.
- All 80 prediction journals passed token, geometry, incumbent, and
  deterministic replay checks.
- The four retention-policy arms used the same spectral candidate construction,
  beam width, greedy-slot count, model, task score, and 2,880-token cap.
- The global CEM arm used the same task, model, action domain, and cap as a
  separate strong comparison.
- No privileged suffix, physical outcome, or posthoc suffix-quality score was
  used for selection.

## Development results

| Arm | Success | Mean terminal distance | Median | Mean prediction tokens |
|---|---:|---:|---:|---:|
| response retention | 2/16 | 0.36928 | 0.50044 | 2879.25 |
| greedy retention | 3/16 | 0.38615 | 0.50044 | 2879.56 |
| diverse retention | 3/16 | 0.31445 | 0.50028 | 2879.25 |
| random retention | 3/16 | 0.38629 | 0.50040 | 2879.31 |
| global CEM | 6/16 | 0.32637 | 0.50047 | 2880.00 |

Relative to response retention, the paired terminal-score comparisons were:

- greedy: 5 wins, 4 losses, 7 ties;
- diverse: 2 wins, 0 losses, 14 ties;
- random: 4 wins, 4 losses, 8 ties;
- global CEM: 8 wins, 4 losses, 4 ties.

The response, greedy, diverse, and random arms each paid the response/SVD and
successor-probe costs. The global CEM arm did not use those computations. The
reported cost is prediction-token cost only; wall time, SVD time, encoding,
memory, synchronization, and GPU cost remain unclosed.

The read-only ledger audit recorded the following aggregate engineering costs:

| Arm | Prediction tokens | Prediction calls | Full-chain samples | SVD calls | Planner wall seconds |
|---|---:|---:|---:|---:|---:|
| response | 46,068 | 1,246 | 13,748 | 160 | 2.420 |
| greedy | 46,073 | 1,268 | 13,771 | 160 | 2.653 |
| diverse | 46,068 | 1,246 | 13,748 | 160 | 2.456 |
| random | 46,069 | 1,247 | 13,751 | 160 | 2.458 |
| global CEM | 46,080 | 1,264 | 15,360 | 0 | 2.423 |

The ledger is available at
`le-wm-main/results/pc_retention_equal_slot_cost_dev_20261104_codex/cost_ledger_audit.json`.
It confirms the recorded token/call/SVD/wall fields, but
`full_cost_certified=false`: GPU identity/utilization/memory, encoding,
preprocessing, synchronization, loading, host memory, and FLOPs were not
measured.

## Interpretation

On this development panel, response retention did not produce a repeatable
advantage over equal-slot alternatives. Diverse retention has the lowest mean
distance, but its advantage comes from only two non-tied paired roots and is
not sufficient for a stable claim. Global CEM is stronger on success count and
paired wins in this panel, so the response computation has not demonstrated
net value after its own token cost.

This result is a bounded negative result for this retention implementation and
task family. It does not distinguish a defective response-value heuristic from
an ineffective retention mechanism in general, and it does not refute the
Planning Compiler theory.

The next required work is not another retention-policy sweep on these roots.
The formal candidate should be frozen or rejected, then evaluated with
complete wall/GPU accounting, PC1 response certification, same-version
mechanism intervention, and the unified RGB/strong-baseline pipeline. PC2 and
PC3 remain open.
