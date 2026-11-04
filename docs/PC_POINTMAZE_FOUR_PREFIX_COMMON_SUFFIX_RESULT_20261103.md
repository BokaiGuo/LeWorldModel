# Four-prefix, five-suffix intervention

The [fixed protocol](PC_POINTMAZE_FOUR_PREFIX_COMMON_SUFFIX_PROTOCOL_20261103.md)
executed 160 physical sequences on eight exposed roots. All terminal
positions matched the separate position backend. The previous 72-cell
panel and the 16 saved greedy/zero own-tail executions reproduced exactly
in terminal RGB and full endpoint; no world-model inference was used here.

Root 1894 was the only root in the preceding panel where the spectral
prefix increased physical goal distance relative to its starting point.
All four prefix states and common suffix outcomes are now available:

| Prefix | cp5 distance | Spectral suffix terminal | Greedy suffix terminal | Zero-action suffix terminal |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 0.6855 | 0.6130 | 0.4590 | 0.6187 |
| Zero | 0.6316 | 0.4358 | 0.2557 | 0.3871 |
| Random | 0.5742 | 0.3234 | 0.3509 | 0.4189 |
| Spectral | 0.5819 | 0.2447 | 0.2053 | 0.3105 |

Against each prefix's own zero-action suffix, the *same* spectral suffix
changes terminal distance by -0.0658 after spectral, -0.0057 after greedy,
and +0.0487 after zero. This indicates a different effect of that suffix
under these saved prefixes, but not independently verified capability gain:
the spectral cp5 state is already closer to the target than the greedy and
zero cp5 states, and the no-action continuation also ends closer. The
greedy-prefix selector itself is model-greedy, not physical-oracle greedy.
The same greedy suffix reaches 0.2053 after spectral versus 0.4590 after
greedy, again with the same starting-state confounding.

All 160 cells, all five suffixes and vector displacement relative to each
prefix's zero-action continuation are saved. The examples are retrospective,
the prefixes were not selected under a predeclared non-progress criterion,
and roots are not independent confirmatory instances. This factorial does
not establish separately optimized capability, total-budget advantage,
independent generalization or P1 admission.

- Report:
  `le-wm-main/results/a1_pointmaze_four_prefix_common_suffix_20261103_DEV01.json`,
  SHA-256 `c028d405d0d5facb3541541f24c59f145511981cf8221138cb5974b006d6693f`.
- Arrays: same basename `.npz`, SHA-256
  `0b44887ac9c899117b2376400b9a5d545cc5cda5a73b8b43c50ab067c16adc51`.
- Saved design, action, replay and outcome arithmetic plus greedy optimizer
  checks: **2 passed**; tests do not rerun physics or neural inference.
