# Saved waypoint suffix cross-prefix replay

All three saved suffixes from each of three exposed search seeds were replayed
after all three fixed prefixes, plus each prefix's zero-suffix control. The
source waypoint methods had access to exact prefix pose and kinematics; this
is an outcome-exposed diagnostic, not an RGB planner or a new method test.

Terminal goal distance, lower is better:

| Seed | Suffix source | Selected prefix | Zero prefix | Progress prefix |
| ---: | --- | ---: | ---: | ---: |
| 20261103 | selected | ~0 | 0.550436 | 0.513514 |
| 20261103 | zero | 0.805331 | ~0 | 0.513514 |
| 20261103 | progress | 0.984592 | 0.645771 | 0.286944 |
| 20261104 | selected | ~0 | 0.544347 | 0.513514 |
| 20261104 | zero | 0.792943 | ~0 | 0.513514 |
| 20261104 | progress | 0.977074 | 0.633039 | 0.252942 |
| 20261105 | selected | ~0 | 0.544477 | 0.513514 |
| 20261105 | zero | 0.720567 | 0.245868 | 0.513514 |
| 20261105 | progress | 0.846486 | 0.630074 | ~0 |

For all three selected-origin suffixes, the selected prefix achieves
near-zero distance without a blocked suffix step; zero incurs six blocked
suffix steps and progress nine. Conversely, the first two zero-origin
suffixes succeed after zero but do not succeed after selected. This shows
that the fixed actions have different effects from the three resulting
states. Their prefix positions are already different: the zero-tail
distances are 1.36, 1.00 and 0.513514, respectively. Consequently the
comparison does not isolate a position-independent control mechanism or
show that selected is easier to plan from. In the separate same-budget
raw-action CEM test selected lost to both controls on all three seeds.

The receipt preserves every trajectory, blocked-step flag, first suffix step
with positive x, and each distance change relative to that prefix's zero
suffix. This run used 144 transitions for source verification, 144 for zero
tails and 432 for the 27 cross-suffix replays. Source candidates were chosen
with outcome-exposed exact dynamics and the three seeds share one task; none
are independent confirmation or physical evidence.

The new replay/binding and tamper tests, together with the related triprefix
tests, report **6 passed**. No gate was changed.

- Protocol: `docs/PC_WALL_WAYPOINT_CROSS_SUFFIX_PROTOCOL_20261103_CODEX.md`
- Runner: `le-wm-main/experiments/spectral_reachability/pc_wall_waypoint_cross_suffix_20261103_codex.py`
- Receipt: `le-wm-main/results/pc_wall_waypoint_cross_suffix_20261103_codex_DEV01.json`
- Tests: `le-wm-main/tests/test_pc_wall_waypoint_cross_suffix_20261103_codex.py`
