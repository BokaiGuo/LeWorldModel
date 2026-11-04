# Feasible-prefix real-feedback physical development result

Protocol: `PC_REACHER_FEASIBLE_PREFIX_PHYSICAL_PROTOCOL_20261102.md`.
Both outcome-exposed roots and all ten predeclared branches completed:
each prefix returned five steps and each replanned suffix returned ten,
without reported termination or truncation. The model and root/goal/history
identity checks passed. All six repeated selected/zero/greedy complete raw
action arrays match the earlier guarded physical report exactly; they are
replays, not independent outcomes.

The table reports evaluator-only qpos distance (lower is better), not native
success. Each arm's suffix was separately optimized from that arm's real
RGB and executed action history with 32 CEM candidate rollouts.

| Root | Arm | Initial | After prefix | Final |
|---|---|---:|---:|---:|
| 20263160 | selected | 1.003500 | 1.037149 | 0.558267 |
| 20263160 | zero | 1.003500 | 1.010031 | 0.546408 |
| 20263160 | same-bank greedy | 1.003500 | 0.982156 | 0.497367 |
| 20263160 | refined 0 | 1.003500 | 1.019245 | 0.557469 |
| 20263160 | refined 1 | 1.003500 | 0.997244 | 0.545422 |
| 20263161 | selected | 1.182600 | 1.267141 | 0.619442 |
| 20263161 | zero | 1.182600 | 1.245444 | 0.598310 |
| 20263161 | same-bank greedy | 1.182600 | 1.148837 | 0.760234 |
| 20263161 | refined 0 | 1.182600 | 1.269222 | 0.711966 |
| 20263161 | refined 1 | 1.182600 | 1.245755 | 0.597585 |

The model selected neither refined candidate. Refined 1's final distance is
approximately 0.000985 and 0.000725 below zero on the two roots respectively,
but is worse than greedy on the first. Its first prefix already reduces
distance from the initial state on the first root. The second root's prefix
is slightly worse than zero after five steps. These very small exploratory
differences are not statistical evidence, a success-rate result, or proof
of task-relevant reconfiguration. The post-hoc best physical candidate is not
the model-selected method. This own-tail comparison cannot establish a
common-suffix capability effect.

All methods use the same suffix optimizer configuration, but upstream
generation and selection costs differ and full encoding/wall-time fairness
is not certified. Stop information is recorded; native success and per-step
qpos trajectories are not exported by this runner and must not be inferred.
The source bank is fixed, so a subsequent intervention can test these
specific prefixes and shared suffixes, but these roots remain development
data and cannot become independent confirmation.

Artifact: `le-wm-main/results/reacher_pc_feasible_prefix_physical_20261102_v1/report.json`.
Five related binding and selection tests passed before execution.
