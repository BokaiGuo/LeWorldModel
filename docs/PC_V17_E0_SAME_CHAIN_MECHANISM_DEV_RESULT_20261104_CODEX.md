# v17 score-augmented response: E0 same-chain mechanism result

The eight prespecified exact-Wall plans and interventions completed. The
same v17 compiler core was used in the separate Reacher feedback
development run, with each task's legal score and chart inputs.

| Predeclared outcome | Roots / attempts |
| --- | ---: |
| Final selected chain distance `<0.05` | 2/8 |
| Selected prefix non-improving | 2/8 |
| Base intersection against zero and same-bank greedy | 1/8 |
| Full intersection also including lookahead and random | 0/8 |

Root 4 is a bounded base-mechanism witness. Initial distance was
1.151043; the selected prefix increased it to 1.184494. The actual
selected suffix then reached 0.038315, a change of −1.146180 relative to
its zero-tail result. The same suffix after zero, greedy, lookahead and
random prefixes ended at 2.005257, 1.841308, 1.637215 and 0.683591.
Under three equal-budget suffix replan streams, mean terminal distances
were 0.006809, 0.501942, 0.500549, 0.500392 and 0.501608 respectively.

The full intersection still fails: the random control changes collision
branch at the stipulated 0.05 finite-difference scale. That predeclared
condition is retained rather than removed after observing the result.
The base witness is not proof of a general score-augmentation effect or
PC3 advantage, and the finite optimization controls are not an upper
bound on ordinary search.

Plan traces were checked using the alternate same-environment geometry;
intervention reconstruction and 68,520 saved selection, replan and common
tail trajectories were also checked. These are same-environment
consistency checks, not independent replication. Eight tasks and one
physical family cannot establish broad generality. Source independence is
unknown; PC2 remains unestablished under the complete predeclared
intersection. No visual or strong-baseline advantage follows.

The eight-task stop has been reached; do not rerun, expand or change the
criteria. Protocol:
`PC_V17_E0_SAME_CHAIN_MECHANISM_DEV_PROTOCOL_20261104_CODEX.md`.
Plans, interventions, raw actions and audits:
`le-wm-main/results/pc_v17_e0_same_chain_mechanism_dev_20261104_codex/`.
