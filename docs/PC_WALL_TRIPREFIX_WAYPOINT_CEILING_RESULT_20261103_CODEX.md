# Triprefix 12-step waypoint ceiling: exposed development result

After the CEM outcomes had been seen, the separately specified diagnostic
reused the three fixed prefixes and three search seeds on the same exact wall
task. Every arm received the same 256 sampled waypoints per seed. The
waypoint constructor uses the exact prefix-end pose and known unicycle
kinematics; it is privileged and is **not** an RGB planner or the D2P
method.

| Seed | Selected error / successes | Zero error / successes | Progress error / successes |
| ---: | ---: | ---: | ---: |
| 20261103 | 7.85e-16 / 1 | 5.66e-16 / 1 | 0.286944 / 0 |
| 20261104 | 4.58e-16 / 3 | 5.55e-16 / 1 | 0.252942 / 0 |
| 20261105 | 4.00e-16 / 1 | 0.245868 / 0 | 2.22e-16 / 2 |

Success counts are candidates with distance below 1e-8 among 256 proposals;
they are not independent episodes or success-rate estimates. Selected-prefix
12-step witnesses reached the goal for all three samples. Zero also has
witnesses in two samples; a progress-prefix witness exists in the third.
Thus the selected prefix's CEM failures cannot be explained simply by an
impossible 12-step suffix on this fixed task. It is consistent with an
optimization/parameterization or information-access limitation, but this
comparison changes the proposal family and gives the waypoint method exact
state and kinematics; it does not isolate either causal factor.

Each arm used 256 exact suffix rollouts / 3,072 state transitions and four
prefix transitions, as in the preceding CEM diagnostic. Constructor overhead,
source-prefix construction and replay are separate; matching dynamics counts
does not match end-to-end compute. These results do not show spectral
superiority, change the old result, or open any gate.

Tests verify bounds, invalid pose rejection, shared sampled waypoints,
selected index, prefix binding and replay; with the related CEM and wall
tests, **8 passed**. The source/receipt hashes and every score and waypoint
are preserved in the result JSON.

- Protocol: `docs/PC_WALL_TRIPREFIX_WAYPOINT_CEILING_PROTOCOL_20261103_CODEX.md`
- Runner: `le-wm-main/experiments/spectral_reachability/pc_wall_triprefix_waypoint_ceiling_20261103_codex.py`
- Receipt: `le-wm-main/results/pc_wall_triprefix_waypoint_ceiling_20261103_codex_DEV01.json`
- Tests: `le-wm-main/tests/test_pc_wall_triprefix_waypoint_ceiling_20261103_codex.py`
