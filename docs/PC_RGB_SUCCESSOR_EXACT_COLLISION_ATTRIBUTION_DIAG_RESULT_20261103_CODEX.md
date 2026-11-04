# Successor exact-wall collision attribution diagnostic

This post-outcome diagnostic follows
`PC_RGB_SUCCESSOR_EXACT_COLLISION_ATTRIBUTION_DIAG_PROTOCOL_20261103_CODEX.md`.
It analyzes the 24 sealed final chains and all 72 recorded response contexts;
it does not rerun the planner or change any selected action.

**20/24 selected chains experienced at least one blocked wall step; all
24 finished on the starting side** (`x < 0`). Among response contexts, 14/72
nominal chains experienced a blocked step. This is consistent with the
observed inability to reach the other side, but collision alone does not
identify whether candidate generation, pruning, optimization, or limited
budget is the cause.

At the original response scale of 0.05, the prior matrix reconstruction was
checked again. Relative to this matrix, changing scale produced:

| Scale | Matrix change >10% | Changed positive/negative collision pattern within scale | Changed recorded collision pattern vs 0.05 |
|---|---:|---:|---:|
| 0.01 | 6/72 | 0/72 | 6/72 |
| 0.1 | 10/72 | 10/72 | 8/72 |

All six >10% changes at 0.01 coincided with a cross-scale pattern change.
At 0.1 all ten >10% changes had at least one different positive/negative
collision pattern within that scale, although only eight also differed from
the 0.05 pattern. These comparisons answer different questions and should
not be conflated. Four reference matrices had norm below `1e-12`; relative
changes use the floor described in the prior diagnostic, whose absolute
errors remain available. Collision switching can explain the presence of
piecewise response behavior in these observations, but this association
does not prove causality or an incorrect wall implementation.

The per-step collision patterns, position replay, matrices' difference
statistics, source and receipt hashes are in
`le-wm-main/results/pc_rgb_successor_exact_same_policy_dev_20261103_codex_collision_attribution_diagnostic.json`.
The related tests passed 10 checks, including independent replay of all 24
selected chains. No PC gate was changed; this is not evidence of PC2
mechanism or PC3 planning advantage.
