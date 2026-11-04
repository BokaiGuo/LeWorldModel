# Shared multistart suffix versus direct search: development result

On the single already exposed exact wall task, each of six model-generated
prefixes received the same two-start, 204-evaluation, 12-step program
optimizer. Direct full-horizon flat and program CEM controls each received
918 evaluations. Scores below are terminal position distance, lower is
better; all selected terminals remain on the starting side of the wall.

| Seed | Executed prefix | Full method | Direct flat | Direct program | Qualified nonprogress prefix |
| ---: | --- | ---: | ---: | ---: | --- |
| 20261103 | zero | 0.500520 | 0.506931 | 0.501181 | none |
| 20261104 | random_1 | 0.500942 | 0.501942 | 0.501449 | random_0, not executed |
| 20261105 | zero | 0.502486 | 0.501999 | 0.502088 | none |

The two signed spectral prefixes' terminal distances were respectively
0.504115/0.512924, 0.511989/0.512947 and 0.503332/0.516138 across the three
seeds; neither was selected or qualified. For seed 20261104, the qualified
random_0 prefix has model prefix cost 0.345123 versus root cost 0.333333,
and its full distance 0.501784 is below both zero and same-bank greedy.
The *actual* selected random_1 has prefix cost 0.273491: it already improves
the immediate task metric and is a different chain. Its outcome cannot be
used as a mechanism witness for random_0. The other two seeds abstain from
a qualified reconfiguration while returning the ordinary best plan.

The six suffix optimizations use 14,688 state transitions per seed;
candidate generation and prefix evaluation use an additional 104, including
one Jacobian and 16 internal finite-difference rollouts. Each direct control
uses 14,688 transitions; thus the method used more total transitions even
before considering SVD, optimizer arithmetic and replay. This is not a
certified equal-total-compute advantage. The parameterized optimizer was
stronger than the earlier discrete Atlas continuation in design, but its
failure on this task does not establish global optimality or rule out a
different optimizer. The three seeds repeat one exposed task, not independent
roots; no real environment or RGB model was evaluated and no gate opens.

The new tests verify bounded programs, first-round paired noise, action
binding and replay, exact budget counts, ordinary selection, distinct
qualified selection and abstention. With the new spectral-feedback component
and primitive-CEM tests, **10 passed** using the project virtual environment.
System Python initially failed collection because `tensordict` was absent;
that environment failure is not a scientific result.

- Protocol: `docs/PC_WALL_SHARED_MULTISTART_PROGRAM_PROTOCOL_20261103_CODEX.md`
- Runner: `le-wm-main/experiments/spectral_reachability/pc_wall_shared_multistart_program_20261103_codex.py`
- Receipt: `le-wm-main/results/pc_wall_shared_multistart_program_20261103_codex_DEV01.json`
- Tests: `le-wm-main/tests/test_pc_wall_shared_multistart_program_20261103_codex.py`

The separate opt-in
`experiments/spectral_reachability/pc_spectral_seeded_feedback_cem_20261103_codex.py`
implements normalized-action chain-rule spectral seeds and the existing
feedback CEM update; its unit tests do not establish integration or physical
performance. It has not yet been connected to the shadow closed-loop runner.
