# Guarded spectral prefixes: common-suffix development result

The pre-execution protocol was
`D2P_PC_REACHER_GUARDED_COMMON_SUFFIX_PROTOCOL_20261102.md`. All 24
branches completed 15 steps. Each original prefix distance replayed within
1e-8 and the root and goal RGB hashes matched. The exact raw action witnesses
are retained in the SHA256-bound input physical report, rather than duplicated
in each result branch; the new report retains stepwise qpos, qvel, reward and
stop status.

| Seed | Common tail | Selected final | Zero final | Selected minus zero | Selected passive-adjusted | Zero passive-adjusted |
|---|---|---:|---:|---:|---:|---:|
| 20263160 | selected | 0.558267 | 0.524961 | +0.033306 | -0.478774 | -0.484008 |
| 20263160 | zero | 0.581415 | 0.546408 | +0.035008 | -0.455626 | -0.462561 |
| 20263160 | greedy | 0.558267 | 0.524961 | +0.033306 | -0.478774 | -0.484008 |
| 20263161 | selected | 0.619442 | 0.598310 | +0.021131 | -0.651643 | -0.647763 |
| 20263161 | zero | 0.619442 | 0.598310 | +0.021131 | -0.651643 | -0.647763 |
| 20263161 | greedy | 0.926983 | 0.895842 | +0.031141 | -0.344102 | -0.350231 |

Distance is evaluator-side qpos distance, lower is better. Passive-adjusted
means final distance under the named tail minus the final distance after the
same prefix with a ten-step zero tail. Thus negative is improvement relative
to that prefix's passive branch, not absolute success.

The named tails are not all meaningfully distinct: at seed 20263160 the
selected and greedy tails differ by at most 8.95e-8 per raw action coordinate;
at seed 20263161 selected and zero differ by at most 1.20e-7. They must not be
treated as independent samples. The selected prefix has a worse absolute
terminal distance than zero for every named nonzero tail. Passive adjustment
is tail-dependent: selected is modestly better only for the near-identical
selected/zero tails at the second root. This does not establish a general
useful future-control advantage and cannot substitute for independent
same-budget replanning or task-success results. These are known development
roots and outcome-exposed tails, with no inferential claim.

The separately replanned own-tail outcomes are reproduced exactly at their
matching intervention branches. This checks the executed action and root
binding for this panel, not the world model's physical predictive accuracy.

Artifact:
`le-wm-main/results/reacher_pc_guarded_common_suffix_development_20261102_v1/report.json`.
