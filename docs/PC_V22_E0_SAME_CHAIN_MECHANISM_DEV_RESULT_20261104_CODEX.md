# v22 reserved-refinement exact-Wall same-chain mechanism result

All eight prespecified plans and interventions completed. Four refinement
starts were used per task, each with 26–28 samples in three batches. A
saved-trace audit reconstructed the elite mean and variance used in every
subsequent batch, confirming actual distribution updates were sampled.
This fixes the specific one-batch budget-allocation limitation observed
in the 1,920-token v21 mechanism panel; it does not establish sufficient
search or imply that the two different panels are a controlled comparison.

| Outcome | Tasks |
| --- | ---: |
| Final chain distance `<0.05` | 0/8 |
| Base zero/greedy mechanism intersection | 1/8 |
| Full zero/greedy/lookahead/random intersection | 1/8 |

On root 5, the selected prefix increased target distance from 1.004988
to 1.223243; the selected suffix ended at 0.189931, a change of −1.033312
relative to its zero-tail result. The same tail after zero, greedy,
lookahead and random prefixes ended at 0.753837, 1.196869, 1.196869 and
1.914943. Mean distances over the three per-prefix replan streams were
0.338961, 0.500258, 0.500209, 0.500209 and 0.500604 respectively.
This is a limited same-chain mechanism witness under the predeclared
intervention, but the final chain did not achieve the `<0.05` task
success threshold; it is not evidence of overall PC2 or PC3 completion.

Plans were sealed before intervention; saved traces and deterministic
reconstruction were audited. Alternate same-environment geometry checked
68,520 saved intervention trajectories. These checks do not certify
independent replication or optimality of the finite control searches.
Exactly eight tasks were attempted; the stop is reached, so do not retry,
replace roots or alter thresholds.

Protocol: `PC_V22_E0_SAME_CHAIN_MECHANISM_DEV_PROTOCOL_20261104_CODEX.md`.
Plans, predictions, interventions and audits:
`le-wm-main/results/pc_v22_e0_same_chain_mechanism_dev_20261104_codex/`.
