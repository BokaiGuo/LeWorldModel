# Block-program wall prospective development result

This is a development experiment, not independent confirmation and not an
admission decision for PC0–PC5. The protocol was written before executing
the 24-task panel: `PC_BLOCK_PROGRAM_WALL_PROSPECTIVE_DEV_PROTOCOL_20261103_CODEX.md`.
The exact wall dynamics provide privileged state and goal information; this is
not an RGB closed-loop experiment.

## Design and accounting

The panel contains 24 predetermined combinations of start position, heading
and goal. Five arms are evaluated on the same roots at 384 and 768 complete
rollouts: SVD response, equivalent ridge response, random response, nominal,
and full-horizon CEM. Each structured arm includes candidate construction,
candidate scoring, block search and unrestricted joint refinement within its
rollout budget. Response extraction uses additional decomposition or solve
operations, so equal rollout counts do not certify equal total compute.

The two budgets reuse the same roots and are not 48 independent tasks.
Planning produced 240 traces and 138,240 scored complete chains. The sealed
audit recomputed all 12,672 recorded batches and checked selection and budget
accounting. The intervention uses additional diagnostic computation, outside
the planner budgets.

## Planning results

Success means final distance below 0.05.

| Arm | 384 mean distance | 384 successes | 768 mean distance | 768 successes |
|---|---:|---:|---:|---:|
| SVD response | 0.49013 | 0/24 | 0.48785 | 0/24 |
| Ridge response | 0.49013 | 0/24 | 0.48785 | 0/24 |
| Random response | 0.47564 | 0/24 | 0.47105 | 0/24 |
| Nominal | 0.45433 | 1/24 | 0.44373 | 1/24 |
| Full CEM | 0.52174 | 0/24 | 0.52172 | 0/24 |

SVD versus nominal paired wins/ties/losses were 6/0/18 at 384 and
11/0/13 at 768. SVD and ridge tied on all roots at both budgets; their maximum
action difference was approximately 6.73e-13. This is consistent with
equivalent regularized full-response solves, not evidence of a separate SVD
benefit. The full-CEM result is one configuration, not a strong-search upper
bound.

The protocol's stop condition was met: the response arm had no successes at
either budget and did not improve mean distance over nominal at either
budget. Pause this six-dimensional response construction under this
protocol; do not select a root or add budgets to override the stop.

## Same-chain mechanism diagnostics

The final selected SVD chains, including any joint refinement, were evaluated
on 48 root-budget pairs. Selected, zero, and same-bank greedy prefixes received
the same predefined suffix family and 120-sample own-prefix replanning.

| Budget | Non-immediate progress | Active response wins both controls | Own replanning wins both | All three on same chain |
|---|---:|---:|---:|---:|
| 384 | 2/24 | 17/24 | 13/24 | 0/24 |
| 768 | 2/24 | 15/24 | 14/24 | 1/24 |

The single 768-budget witness is root 22. Initial distance was 1.280625,
distance after the selected prefix was 1.361008, and final selected-chain
distance was 0.500778: the chain did **not** succeed. For the actual selected
suffix, changes relative to each prefix's zero suffix were -0.860230,
0.587971, and 0.087051 for selected, zero, and greedy respectively. The
corresponding independently optimized distances were 0.526028, 0.686797,
and 0.536440. The selected-versus-greedy margin was about 0.0104.

This is a limited same-chain development signal, not a PC2 pass, a successful
trajectory, a planning advantage, or independent evidence. The panel has
already been exposed and the stopped construction must not be tuned on this
root.

## Feasibility and provenance

After planning, privileged analytic trajectories were replayed as a separate
feasibility diagnostic. All 24 tasks had at least one witness within 15
steps, with 46 witnesses across both sides. These trajectories were not
provided to the planner. Their existence does not establish that a
budget-limited optimizer must find them.

Machine-readable plans, audit, intervention, and feasibility records are in
`le-wm-main/results/pc_block_program_wall_prospective_dev_20261103_codex_*.json`.
Records bind the protocol and implementation hashes. The current experiment
does not establish development-external, normalizer-fit-external, or
checkpoint-training-external provenance; it cannot confirm PC3, PC4, PC5,
or the original spectral-increment contract.
