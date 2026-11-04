# Common-suffix intervention on feasible development prefixes

The fixed five-prefix by six-suffix intervention was executed on each of the
two known roots: all 60 branches completed 15 steps without reported stop.
Every first-prefix distance replayed within 1e-8 of the preceding physical
run. The ten own-tail branches match their original final distances within
5e-7; the intervention reuses saved actions cast to float32, so this is a
numerical tolerance, not bitwise identity.

The table shows refined 1 minus zero in absolute final qpos distance for the
same tail (negative favors refined 1), and the corresponding difference in
each arm's change relative to its own passive-zero tail.

| Root | Common suffix | Absolute delta | Passive-adjusted delta |
|---|---|---:|---:|
| 20263160 | selected | -0.021470 | -0.006874 |
| 20263160 | zero | -0.023374 | -0.008778 |
| 20263160 | same-bank greedy | -0.021470 | -0.006874 |
| 20263160 | refined 0 | -0.023374 | -0.008778 |
| 20263160 | refined 1 | -0.027598 | -0.013002 |
| 20263161 | selected | -0.000725 | -0.001151 |
| 20263161 | zero | -0.000725 | -0.001151 |
| 20263161 | same-bank greedy | +0.004757 | +0.004331 |
| 20263161 | refined 0 | +0.002154 | +0.001728 |
| 20263161 | refined 1 | -0.000725 | -0.001151 |

The first root's refined-1 prefix has already reduced qpos distance from
initial (0.997244 vs 1.003500), so its common-tail advantage cannot by itself
establish a no-immediate-progress mechanism. On the second root, results are
tail-dependent and tiny; the prefix is slightly worse than zero at step 5.
Both roots and all tails were exposed during development. Selected remained
the original candidate; post-hoc refined-1 outcomes are not method outcomes.
No significance or independent-sample claim is warranted.

There are near-identical tails: on root 20263160 selected/greedy differ by at
most 8.95e-8 per action coordinate and zero/refined-0 are identical; on
20263161 selected/refined-1 are identical and zero/refined-1 differ by at
most 1.20e-7. Thus the 60 branches do not represent 60 independent action
experiments. Common-tail effects cannot replace each-prefix same-budget
planning, strong baselines, independent roots, or native task success.

Artifact: `le-wm-main/results/reacher_pc_feasible_common_suffix_20261102_v1/report.json`.
