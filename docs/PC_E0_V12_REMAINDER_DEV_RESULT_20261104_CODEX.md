# E0 v12 remaining-batch repair: development result

## Code issue and repair

v11 allocated its remaining whole-chain prediction budget only in complete
batches of 12. Consequently, it could leave up to 11 full-chain evaluations
unused despite sufficient token budget. This is a confirmed search-budget
utilization defect, not proof that it caused the planning failures.

The new v12 keeps v11's existing complete batches and final paired-comparison
reserve. It uses one partial batch for remaining whole-chain evaluations,
without duplicating the separately tracked incumbent in that batch. It still
preserves the best complete chain. v11 and its sealed results were not changed.
The new implementation is a development candidate: its RGB worker has **not**
yet been switched or certified.

The original v11 tests plus v12 regression tests passed: **20 tests**. Tests
cover 5/10/15-step horizons, prediction-token accounting, preservation of all
v11 predictor batches before the final pair, partial batch sizing, incumbent
non-regression and a single-sample remainder. These are software checks, not
physical-control or gate evidence.

## Fixed E0 comparison

All 24 prespecified plans finished. Scores are terminal Euclidean distances;
success means distance `<0.05`.

| Arm | Success | Mean terminal distance | Mean prediction tokens |
|---|---:|---:|---:|
| v11 | 0/8 | 0.501253 | 1908.50 |
| v12 | 0/8 | 0.501253 | 1919.75 |
| Three-start direct CEM | 0/8 | 0.502262 | 1920 |

v12 used respectively 5, 1, 1, 1, 11, 9, 1 and 1 additional full-chain
evaluations across the eight roots, but the selected score was identical to
v11 on every root. The matched original batches were bitwise identical; the
independent replay audit checked all 16,990 scored trajectories, stored scores,
action bounds, packet accounting, and final incumbent selection. One v12 run
left two tokens, insufficient for another three-packet full chain.

The panel is stopped as specified. These exposed development roots are not
certified independent of all prior experiments. Equal token caps do not imply
equal total computation, and the baseline is not a certified strong-search
upper bound. The defect is repaired in v12, but the experiment provides no
evidence that this repair resolves automatic discovery, PC2, or PC3.

## Artifacts

- Protocol: `docs/PC_E0_V12_REMAINDER_DEV_PROTOCOL_20261104_CODEX.md`
- Implementation: `le-wm-main/experiments/spectral_reachability/pc_recursive_response_v12.py`
- Tests: `le-wm-main/tests/test_pc_recursive_response_v12.py`
- Full traces, results, dependency manifest and audit:
  `le-wm-main/results/pc_e0_v12_remainder_dev_20261104_codex/`

The audit's independent geometry shares the mathematical E0 environment and
is not an independent scientific replication. No changes are made to the
original gate thresholds or to previous negative results.
