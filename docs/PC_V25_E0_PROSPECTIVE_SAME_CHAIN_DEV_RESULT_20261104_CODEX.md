# v25 prospective same-chain E0 mechanism: development result

All eight predeclared plans were sealed and audited before the mechanism
interventions. All eight interventions completed. The actual final chain
after joint refinement and paired abstention was used for every test;
none of the eight plans abstained. The existing `<0.05` task-success
threshold was not reached by any plan.

| Root | Selected final distance | Selected prefix + zero tail | Selected prefix + selected tail | Nonprogress | Base intersection | All-control intersection |
| --- | ---: | ---: | ---: | --- | --- | --- |
| 0 | 0.503515 | 0.742514 | 0.503515 | no | no | no |
| 1 | 0.523425 | 0.772030 | 0.523425 | no | no | no |
| 2 | 0.506315 | 0.718686 | 0.506315 | no | no | no |
| 3 | 0.506196 | 0.660935 | 0.506196 | no | no | no |
| 4 | 0.505022 | 0.660935 | 0.505022 | no | no | no |
| 5 | 0.504871 | 0.657968 | 0.504871 | no | no | no |
| 6 | 0.511387 | 0.563564 | 0.511387 | no | no | no |
| 7 | 0.502290 | 0.742514 | 0.502290 | no | no | no |

All selected prefixes were nonzero, but **0/8 satisfied the original
nonprogress predicate**. The base intersection against zero and greedy
was 0/8; the full intersection against zero, greedy, lookahead and
random bridge was also 0/8. These are not eight independent successes
for an alternative definition: no threshold was changed.

The selected prefix was exactly the same as the same-bank greedy prefix
on 5/8 roots. It had a strictly better selected-tail improvement relative
to each prefix's own zero tail on 7/8 versus zero, 3/8 versus greedy, 6/8
versus lookahead and 5/8 versus random. The corresponding equal-budget
replanning mean was strictly better on 7/8, 2/8, 5/8 and 8/8 respectively.
Response-matrix and goal-gradient gaps exceeded the existing thresholds
at both finite-difference scales on 8/8, 3/8, 8/8 and 8/8 respectively.
The selected control's branch-stability predicate at both scales held on
0/8. Component counts overlap and do not replace the intersection.

Each root used at most 1,920 prediction tokens for planning. The
post-seal mechanism diagnostic separately recorded 854 endpoint calls,
9,870 samples and 29,592 five-step prediction tokens per root; local
branch replays are additional computation. Its complete endpoint
journals and final-chain bindings were checked against an alternate
implementation of the same geometry. Intervention arrays and decisions
were also reconstructed using the same implementation. Neither check is
an independent replication.

This prospectively evaluates the actual v25 selected chains on this
development panel and does not establish the PC2 mechanism. Broader root
provenance is unknown; the result does not prove that the theory is false.
The panel is stopped; do not adjust predicates, replace failures or reopen
roots. PC3 performance and RGB control remain separate.

Protocol: `PC_V25_E0_PROSPECTIVE_SAME_CHAIN_DEV_PROTOCOL_20261104_CODEX.md`.
Raw records, prediction journals, `plans_audit.json`,
`interventions_audit.json` and `interventions_geometry_audit.json`:
`le-wm-main/results/pc_v25_e0_prospective_same_chain_dev_20261104_codex/`.
