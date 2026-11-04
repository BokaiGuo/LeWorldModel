# Wall triprefix CEM and common-suffix diagnostic

The frozen post-outcome development protocol was executed on the one exposed
exact wall task. The selected spectral prefix moves directly backward; its
prefix-end distance is 1.36, compared with 1.00 for zero and 0.513514 for
progress. This is not a learned-model or independent physical experiment.

| Seed | Selected, own CEM | Zero, own CEM | Progress, own CEM |
| ---: | ---: | ---: | ---: |
| 20261103 | 0.539872 | 0.507672 | 0.504946 |
| 20261104 | 0.508095 | 0.503182 | 0.502832 |
| 20261105 | 0.512045 | 0.502647 | 0.502779 |

All values are terminal goal distances (lower is better). The selected prefix
loses to both controls under this finite CEM budget in all three repeated
searches, and no selected trajectory crosses the wall. The seeds are not
independent tasks. This does not prove that all optimizers or suffix budgets
would fail.

The following uses the *selected arm's* optimized suffix on every prefix:

| Seed | Selected terminal / change | Zero terminal / change | Progress terminal / change |
| ---: | ---: | ---: | ---: |
| 20261103 | 0.539872 / -0.820128 | 0.584714 / -0.415286 | 0.524826 / +0.011313 |
| 20261104 | 0.508095 / -0.851905 | 0.507525 / -0.492475 | 0.546275 / +0.032762 |
| 20261105 | 0.512045 / -0.847955 | 0.573013 / -0.426987 | 0.559249 / +0.045736 |

Change is relative to that prefix followed by a zero suffix, not to a shared
starting distance. The selected suffix was optimized for the selected state;
its larger reduction from a worse starting point does not establish better
controllability. The common-suffix and own-optimizer columns answer distinct
questions and must not be substituted for one another.

Each arm performed 256 suffix rollouts / 3,072 exact transitions; each prefix
was separately rolled out for four transitions. The nine optimized arms
therefore used 27,648 suffix and 36 prefix transitions, excluding selected
replays and common/zero-tail evaluations. Candidate-generation, original
spectral Jacobian work, and runtime are not matched as a full method budget.
The same RNG seed gives paired base normal draws per arm, not identical
adaptively transformed suffixes. All selected suffixes and replayed
trajectories are in the receipt.

Prefix binding, action limits, deterministic CEM, costs and common-tail
binding were checked along with existing wall tests: **9 passed**. These
tests and the exposed analytical result do not open a gate.

- Protocol: `docs/PC_WALL_TRIPREFIX_CEM_INTERVENTION_PROTOCOL_20261103_CODEX.md`
- Runner: `le-wm-main/experiments/spectral_reachability/pc_wall_triprefix_cem_intervention_20261103_codex.py`
- Receipt: `le-wm-main/results/pc_wall_triprefix_cem_intervention_20261103_codex_DEV01.json`
- Tests: `le-wm-main/tests/test_pc_wall_triprefix_cem_intervention_20261103_codex.py`
