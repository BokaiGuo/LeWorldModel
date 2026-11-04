# Goal-Q versus latent L2 on saved same-action outcomes

Under `PC_REACHER_LATENT_Q_VS_L2_PROTOCOL_20261102.md`, the saved 70-branch
RGB replay was compared with the separately re-encoded goal embedding.
For each root the table counts reversals in the same 30 prefix-versus-neutral
contrasts, each using the same tail. A reversal means strictly opposite
signs; there were no exact ties.

| Root | Cost | Predicted vs actual-RGB reversal | Actual-RGB vs qpos reversal |
| --- | --- | ---: | ---: |
| 20263160 | Goal-Q | 1/30 | 14/30 |
| 20263160 | Latent L2 | 7/30 | 10/30 |
| 20263161 | Goal-Q | 0/30 | 9/30 |
| 20263161 | Latent L2 | 10/30 | 8/30 |

On these exposed roots the unweighted metric has fewer RGB-to-qpos reversals,
but more prediction-to-RGB reversals. Both exhibit both kinds of mismatch;
the differences do not establish that replacing Q improves action selection
or physical closed-loop control. The 30 comparisons within each root share
histories and actions, so they are not 30 independent trials. Q and L2 are
different latent metrics and the qpos distance is an evaluator-only task
metric; no simulator state is fed to a planner.

No neural inference, new physical execution, action selection or gate change
occurred. The result is exploratory and outcome-exposed; any planner change
requires a new fixed development protocol and independent validation.
The detailed per-pair deltas are in
`le-wm-main/results/reacher_pc_latent_q_vs_l2_20261102_v1.json`, SHA-256
`53a18604666c2731a5a8c4926f06834e12e11908648ee1704f1c259ca5318376`.
