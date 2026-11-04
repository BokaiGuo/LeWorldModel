# Exact wall-detour development result

The fixed protocol is `PC_EXACT_WALL_KILL_TEST_PROTOCOL_20261102.md`. Four
tests passed, including wall crossing, endpoint collision, action bounds and
replay of a geometrically feasible witness.

| Planner | Terminal distance | Max distance to goal | Simulated transitions | Scores |
| --- | ---: | ---: | ---: | ---: |
| One-step greedy | 0.585640 | 1.000000 | 204 | 204 |
| Full-horizon CEM | 0.500418 | 1.037645 | 24,576 | 2,048 |
| Beam search | 0.511414 | 1.000000 | 3,706 | 3,706 |
| Geometric witness | < 1e-15 | 1.208305 | 12 | 0 |

The wall separates start and goal, and crossing the wall requires |y|>=1;
the point of crossing must therefore be at least sqrt(1.25) from the goal.
Since initial distance is 1, any valid trajectory reaching the other side
must at some point move farther from the goal. The known-geometry witness
crosses at y=1.1, reaches the goal and uses no rejected transition. Its
construction has privileged task knowledge and is not a fair learned-model
baseline.

The three tested generic searches do not reach the target at their fixed
settings. This is one deterministic, manually specified task with differently
accounted operations, not a benchmark of statistical success or equal total
compute. The hard collision transition is nonsmooth; gradient/GN and random
bridge baselines are not yet included. No D2P spectral program was tested,
and this point robot's free-space action Jacobian is constant, so the result
does not establish a future-control-capacity change or P1. It does provide a
valid detour stress case for subsequent optimization and timing comparisons.

The complete programs and trajectories are in
`le-wm-main/results/pc_exact_wall_kill_development_20261102_v1/report.json`.
