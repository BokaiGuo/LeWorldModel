# Score-augmented response: Reacher RGB feedback development result

The frozen-model three-arm experiment completed one prespecified Reacher
task, with three real RGB feedback decisions per arm. The score-augmented
v17, chart-only v13 and shared-bank per-step CEM used the same initial
context and 1,920 prediction tokens at each 15/10/5-step decision.

| Arm | Initial physical distance | After 5 actions | After 10 actions | After 15 actions |
| --- | ---: | ---: | ---: | ---: |
| v13 | 0.763340 | 0.873154 | 0.776259 | 0.023511 |
| v17 score augmented | 0.763340 | 0.373143 | 0.281498 | 0.029644 |
| Shared-bank CEM | 0.763340 | 0.806942 | 0.524104 | 0.021513 |

On this task v17 moved closer sooner, but had the largest final physical
distance of the three. Its final model MSE was lower, but each arm reached
a different actual observation context; those model scores are not a
same-context physical superiority comparison. One task cannot establish
a general advantage or a statistically meaningful effect.

The saved-data audit checked full-chain incumbents, action bounds, exact
token counts, recorded carry, decision hashes and chronological dispatch.
A separate audit reconstructed the bank for all three arms and verified
the paired zero-prefix comparison. Same-backend physical replay reproduced
the saved RGB/history contexts and physical distances exactly. These are
implementation consistency checks, not independent replication or proof
of absence of privileged access: evaluation and planning ran in the same
process. Prediction tokens do not certify total cost matching.

The prespecified single-task stopping rule has been reached: do not retry,
replace this root or tune the configuration on it. The observed 15-step
trajectories provide feasibility witnesses for this particular task;
provenance outside the inspected records remains unknown. This result
does not establish PC2, PC3, long-horizon PC4 or PC5.

Protocol: `PC_V17_REACHER_SCORE_RESPONSE_DEV_PROTOCOL_20261104_CODEX.md`.
Manifest, raw decision traces, physical trajectories, saved-data audit,
bank/pair audit and replay:
`le-wm-main/results/pc_v17_reacher_score_response_dev_20261104_codex/`.
