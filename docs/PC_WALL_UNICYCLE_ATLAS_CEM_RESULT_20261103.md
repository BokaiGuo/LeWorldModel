# Wall-unicycle search development results

The exact analytical wall task has orientation-dependent action response and
a privileged valid 16-step witness. The witness reaches the target within
`3e-16`, with no blocked translation and maximum endpoint goal distance
1.2083 versus initial distance 1.0. A successful continuous crossing has to
leave the initial unit-distance circle. This demonstrates feasibility and a
detour requirement, not a real-world reconfiguration mechanism.

| Seed | Atlas terminal error | CEM terminal error | Post-pilot random waypoint error |
| ---: | ---: | ---: | ---: |
| 20261103 | 0.505090 | 0.550612 | 5.6e-17 |
| 20261104 | 0.503323 | 0.508483 | 0 |
| 20261105 | 0.502205 | 0.511999 | 1.1e-16 |

All Atlas and CEM selected trajectories end on the start side of the wall.
Atlas used 3,320 state transitions per run: 316 compiler rollout calls and
31 Jacobians, the latter including 496 internal finite-difference rollouts.
CEM used 3,264 transitions with 204 full-program evaluations per run.
The post-pilot waypoint control used 3,312 transitions and 207 complete
program evaluations; selected-program replays are separate. Its selected
trajectories had no blocked translation. There were 21, 23 and 17 waypoint
programs with error below `1e-8`, respectively.

The waypoint control uses exact known start/goal pose and a manually fixed
task-scale waypoint box selected after the original pilot. It does not inspect
wall geometry in program construction, but it is neither an RGB planner nor
an untuned general-purpose baseline. Matching only transition counts does not
match wall time, model calls, Jacobian arithmetic or total compute. The
three seeds are repeated searches on one problem, not independent tasks.
The selected results demonstrate a search-coverage / optimization problem
in these implementations; they do not establish a failure of a learned world
model or the underlying planning-compiler hypothesis. Atlas's finite
differences at collision boundaries are nonsmooth and are not certified as
ordinary derivatives there. A subsequent instrumented replay of all three
Atlas searches found **0 differing collision-branch pairs among 744
finite-difference coordinates** (248 per seed), and maximum absolute
Jacobian entry about 1.0. It exactly reproduced the saved selected programs
and compiler counts. Thus this specific branch-crossing mechanism was not
observed at sampled points; that does not prove the derivatives correct or
exclude finite-amplitude prediction and pruning problems. A boundary
counterexample verifies that the trace detects branch changes when they
occur. Audit replay consumed another 9,960 transitions.

Unit checks validate action bounds, blocked translation with independent
heading change, witness replay, FD parity away from the wall, selection and
budget checks. Relevant regression set including the FD audit: **18 passed**.

- Pre-run Atlas/CEM protocol:
  `docs/PC_WALL_UNICYCLE_ATLAS_CEM_PROTOCOL_20261103.md`
- Post-pilot waypoint protocol:
  `docs/PC_WALL_UNICYCLE_WAYPOINT_CONTROL_PROTOCOL_20261103.md`
- Atlas/CEM results:
  `le-wm-main/results/pc_wall_unicycle_atlas_cem_20261103_DEV01.json`,
  SHA-256 `7f7136d1f4192f79e680d10ef79fb06ee76dd5390daef87fc5227a40a85a122d`
- Waypoint results:
  `le-wm-main/results/pc_wall_unicycle_waypoint_control_20261103_DEV01.json`,
  SHA-256 `ea1462b748b17663711ceda0da77606f9ee98e209c92dd24d06c1f51af897570`
- FD audit protocol and result:
  `docs/PC_WALL_UNICYCLE_FD_BRANCH_AUDIT_PROTOCOL_20261103.md` and
  `le-wm-main/results/pc_wall_unicycle_fd_branch_audit_20261103_DEV01.json`
