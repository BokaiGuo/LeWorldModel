# Reacher spectral-prefix common-suffix development result

The pre-execution protocol is in
`D2P_PC_REACHER_SPECTRAL_COMMON_SUFFIX_PROTOCOL_20261102.md`. All 24
branches completed all 15 physical steps. All original prefix qpos distances
replayed within the specified 1e-8 tolerance; root and goal RGB hashes were
checked. Full qpos, qvel, reward and stopping information are retained in the
result JSON. These are two known development roots.

For both roots the separately planned selected and zero tails were exactly
equal as saved raw actions. Thus the four named suffixes contain only three
distinct sequences: zero, the shared selected/zero tail and the greedy tail.
They must not be counted as four independent suffixes.

| Root | Shared ten-step tail | Selected final | Zero final | Selected minus zero | Selected change vs own passive | Zero change vs own passive |
|---|---|---:|---:|---:|---:|---:|
| 20263160 | selected/zero | 0.548603 | 0.546408 | +0.002196 | -0.462848 | -0.462561 |
| 20263160 | greedy | 0.733383 | 0.730984 | +0.002398 | -0.278068 | -0.277984 |
| 20263161 | selected/zero | 0.599019 | 0.598310 | +0.000708 | -0.647901 | -0.647763 |
| 20263161 | greedy | 0.716593 | 0.715992 | +0.000601 | -0.530326 | -0.530081 |

The metric is evaluator-side qpos distance, where smaller is better. The
passive-adjusted change is final distance under the specified tail minus final
distance after the same prefix and ten zero actions; negative means improvement
relative to that prefix's passive branch. Selected has slightly more negative
passive-adjusted changes in these four comparisons, but its absolute final
distance is worse than zero in all four. The size of the adjusted differences
is 0.00008–0.00029, with no uncertainty estimate, no independent roots and no
task-success evaluation. This is not evidence of useful reconfiguration or a
controllability advantage; nor does it exclude such an effect elsewhere.

Previously planned own-tail outcomes are exactly recovered by the respective
common-tail branches, which checks a concrete replay and action-binding issue.
The intervention does not establish that the model predicted the physical
response correctly; the experiment did not encode the actual successor RGB
and compare predicted and observed effects.

Artifact:
`le-wm-main/results/reacher_pc_spectral_common_suffix_development_20261102_v1/report.json`.
