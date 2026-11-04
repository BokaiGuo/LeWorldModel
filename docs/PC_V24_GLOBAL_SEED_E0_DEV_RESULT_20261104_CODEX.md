# v24 global-seed exact-Wall development result

All 24 plans completed. The same-environment trace audit checked common
proposal banks, scored actions, terminal selection, and prediction-token
accounting. A separate seed-use audit confirmed that the second global
proposal entered the actual refinement seed batch on all eight v24 roots;
it received 29–33 scored samples per root.

| Root | v23 distance | v24 distance | shared CEM distance |
| --- | ---: | ---: | ---: |
| 0 | 0.505724 | 0.505724 | 0.502015 |
| 1 | 0.501479 | 0.501479 | 0.055154 |
| 2 | 0.170236 | 0.170236 | 0.056274 |
| 3 | 0.503091 | 0.510951 | 0.500267 |
| 4 | 0.509712 | 0.505242 | 0.500756 |
| 5 | 0.436700 | 0.505258 | 0.501627 |
| 6 | 0.504544 | 0.513624 | 0.500940 |
| 7 | 0.507937 | 0.507477 | 0.502556 |

All three arms succeeded on **0/8** at the frozen 0.05 threshold. Compared
with v23, v24 was better on two roots, worse on three, and tied on three.
It did not establish the hypothesized improvement. Equal overall tokens do
not imply equal refinement budgets; the added start also displaces another
refinement seed. This is not a clean isolated estimate of seed quality.

The result is restricted to this new E0 development panel; source
independence is unknown. The audits use recorded traces and the same
environment rather than independent replication. This panel is stopped.
No PC2 or PC3 gate follows from it. Raw results are in
`le-wm-main/results/pc_v24_global_seed_e0_dev_20261104_codex/`, and the
seed-use audit is
`le-wm-main/results/pc_v24_global_seed_use_audit_20261104_codex.json`.
