# Goal-encoder co-metric on sealed feasible-prefix trajectories

The predeclared retrospective diagnostic
`PC_REACHER_FEASIBLE_GOAL_COMETRIC_PROTOCOL_20261102.md` completed on host
CUDA for both known roots. Goal RGB, root RGB, sampled history index, frozen
checkpoint and normalization hashes matched. Each goal used 192 encoder input
VJPs and 13 metric encoder forwards; all six directional finite-difference
checks per goal passed, with maximum relative errors 0.000986 and 0.000647.
The separate initial goal encoding and other preprocessing costs are not
included in those metric forward counts.

The table counts pairwise ranking reversals against evaluator-only qpos
distance across the same five physical arms at each time point. Strict
opposite signs are counted, with no tolerance; each row has ten pairs.

| Root | Step | Actual latent Euclidean vs qpos | Actual latent Q vs qpos |
|---|---:|---:|---:|
| 20263160 | 5 | 5 | 6 |
| 20263160 | 10 | 0 | 0 |
| 20263160 | 15 | 1 | 4 |
| 20263161 | 5 | 2 | 3 |
| 20263161 | 10 | 4 | 1 |
| 20263161 | 15 | 2 | 1 |

For the particular fixed selected-minus-refined-1 contrast at step 15,
Q-scored open prediction, Q-scored actual RGB latent and physical qpos
distance are respectively +0.001437, +0.002042 and +0.012845 on root
20263160, and +0.009413, +0.006185 and +0.021856 on root 20263161.
Positive values favor refined 1 for all three costs. This differs from the
Euclidean comparison, but the complete table does not show uniformly better
Q-to-physical ordering.

These are same-time score comparisons on already executed and exposed action
chains, whose individual suffixes were optimized after real feedback. They do
not reconstruct the scores of all selector-time imagined candidates, change
the action previously selected, establish an equal-budget planner, or prove
the physical mechanism. Pair counts share roots and trajectories; no
independent statistical or success-rate claim is supported. The Q construction
uses only goal RGB and the frozen encoder, but it remains a task proxy, not
a privileged-state score or validated physical value function.

Artifact: `le-wm-main/results/reacher_pc_feasible_goal_cometric_20261102_v1/report.json`.
Score and replay regression tests passed 3 cases; these tests do not measure
physical efficacy.
