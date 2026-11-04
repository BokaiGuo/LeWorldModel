# Full-horizon wall response: development result

Completed all eight prospectively fixed start-height / heading / goal-height
combinations, six arms, and two model-call caps (96 plans). Existing gates,
shared planners, and historical outcomes were not edited.

## Complete panel

| Arm | Mean terminal distance, 512 calls | Mean terminal distance, 1024 calls |
| --- | ---: | ---: |
| Task-ranked one-mode initialization | .499787 | .503618 |
| Gain-ranked one-mode initialization | .500490 | .504602 |
| Full response, no-SVD ridge | .506811 | .441277 |
| Random initialization | .507707 | .503979 |
| Flat zero-initialized CEM | .509593 | .505077 |
| State-blind primitive CEM | .490186 | .498992 |

All arms had 0/8 endpoints below the predeclared .05 distance threshold at
both budgets. Do not promote small wall-side distance differences as task
success. In particular, all selected plans in the high-budget task/gain
arms remained on the starting side of the wall.

Selected-chain wall crossings (ever x>0), not success:

| Arm | 512 calls | 1024 calls |
| --- | ---: | ---: |
| Task | 1 | 0 |
| Gain | 1 | 0 |
| Ridge | 0 | 2 |
| Random | 0 | 0 |
| Flat | 0 | 0 |
| Primitive | 2 | 1 |

The best high-budget ridge endpoint distance was .0989054, still outside
the threshold. These are counts over selected trajectories only, not the
number of crossing candidates generated during search.

Post-selection inspection of all saved response matrices found rank two.
Thus this failure is not explained by the earlier rank-one mode-selection
degeneracy. Rank alone does not establish local model validity across
collision boundaries, nor the existence of useful detour proposals.

## Feasibility, resource and engineering checks

Independent privileged geometric paths reach every target in 11–14 actions
with no blocked transitions and terminal error below 6e-16. These paths are
saved separately, are never passed into initialization/selection, and only
certify exact-task feasibility. They do not prove an RGB model can plan.

Each arm consumes exactly its declared 512 or 1024 full-chain evaluations,
8192 or 16384 dynamics transitions. Response generation costs 264 evaluations;
the other arms start with four. SVD and linear solves are recorded separately.
This equal transition accounting does not establish equal wall time/FLOPs.
Budgets use different CEM population sizes and are not nested sample sets;
a worse high-budget result is not by itself an accounting bug.

The common optimizer searches full actions for task/gain/ridge/random/flat.
The primitive arm uses the same CEM engine but a four-dimensional action
parameterization and is a separate representation control. None is a new
hierarchical compiler. Feasibility checks and selected-chain replays are
outside planning budgets.

Tests cover continuous batched trajectories, wall blocking, action bounds,
exact budget exhaustion, best-incumbent preservation, ridge without SVD,
deterministic replay without access to the feasibility generator, all 96
saved action/trajectory bindings, source hashes and summary recomputation.
These checks reduce specific engineering uncertainties; they do not certify
all code correct or attribute failure uniquely to theory or implementation.

## Research decision

The new task family avoids easy-task saturation, but the current local
terminal-response initialization does not solve it. Do not freeze it for
visual or independent confirmation. The actionable gap is discovery and
retention of detour-reaching partial programs plus useful continuation,
not another change from rank-one to rank-two SVD.

A subsequent protocol should isolate partial-program coverage, pruning and
continuation with a common suffix optimizer and strong no-SVD/random controls.
Any successful-path injection must remain a separate representability/debug
diagnostic, never a method result. Selected-chain mechanism interventions
remain necessary even if endpoint performance later improves.

No RGB execution, native simulator success, independent confirmation, or
P1 admission is claimed.

Artifacts:

- `docs/PC_WALL_RESPONSE_SIDE_PROTOCOL_20261103.md`
- `le-wm-main/experiments/spectral_reachability/pc_wall_response_side_20261103.py`
- `le-wm-main/results/pc_wall_response_side_20261103.json`
- `le-wm-main/tests/test_pc_wall_response_side_20261103.py`
