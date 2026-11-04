# Fixed feedback-tail prediction versus execution

For each of the 32 saved rejected-prefix trajectories, the historical action
normalizer receipt and final CEM candidate bank reproduce the ten physical
actions actually executed after cp7. The scores below compare those same
chosen actions' predicted terminal scores with observed cp17 outcomes.
Each comparison is within root, across eight different cp7 contexts.

| Root | Predicted vs actual RGB cost | Predicted vs physical distance | Actual RGB cost vs physical distance |
| ---: | ---: | ---: | ---: |
| 8343 | 4/28 | 5/28 | 1/28 |
| 8345 | 4/28 | 5/28 | 0/28 |
| 8349 | 2/28 | 2/28 | 0/28 |
| 8351 | 3/28 | 4/28 | 1/28 |

Values are pairwise ranking reversals. Root 8345 has one tie in actual RGB
cost; ties are not reversals. Each pair shares the same root and must not be
treated as an independent experimental unit. Physical distance and the
projected RGB-derived cost are different quantities; the former never
entered planning. The latter uses estimated state encoded through the proprio
path, not pure visual-token distance.

This supports a real prediction-ranking discrepancy even when the evaluated
and executed suffix actions match. It does not isolate the causes between
the world model, state estimation, observation timing, or remaining interface
contracts, nor does it measure an independent closed-loop advantage. The
check reconstructs scores and actions from saved receipts, without rerunning
model inference; training and normalizer provenance remain uncertified.

Audit script:
`le-wm-main/experiments/pc_causal_mixed_same_tail_audit_20261103.py`.
Machine-readable result:
`le-wm-main/results/pc_causal_mixed_same_tail_audit_20261103_DEV01.json`.
