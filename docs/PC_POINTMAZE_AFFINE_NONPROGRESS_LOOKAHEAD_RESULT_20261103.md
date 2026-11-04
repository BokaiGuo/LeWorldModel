# Affine-score nonprogress lookahead on obstacle roots

The [development protocol](PC_POINTMAZE_AFFINE_NONPROGRESS_LOOKAHEAD_PROTOCOL_20261103.md)
applied the analytic affine score to the existing nine-prefix banks on four
previously exposed obstacle roots. For every prefix the frozen model scored
50 two-packet samples with the same CEM settings. An eligible nonzero
prefix had to show predicted non-improvement at cp5 and a strictly better
optimized terminal score than both zero and model-greedy controls.

| Root | Eligible indices | Best eligible predicted terminal | Zero predicted terminal | Greedy predicted terminal | Selected |
| ---: | --- | ---: | ---: | ---: | --- |
| 8343 | 1, 3, 6, 8 | 1.2842 | 1.0614 | 0.5291 | None |
| 8345 | 1, 3, 6, 8 | 2.0302 | 1.6248 | 0.8568 | None |
| 8349 | 2, 4, 6, 8 | 1.2616 | 1.3126 | 0.7391 | None |
| 8351 | 2, 4 | 2.2820 | 2.2251 | 1.0398 | None |

No prefix passed the model-side selection rule. The greedy and zero arms
were nevertheless each executed on all roots; after five physical actions
each replanned its own five-action tail from actual cp4/cp5 RGB and causal
velocity with the same score, CEM seed and 50-sample budget:

| Root | Greedy cp5 / cp10 physical distance | Zero cp5 / cp10 physical distance |
| ---: | ---: | ---: |
| 8343 | 0.9863 / 0.1071 | 1.4333 / 1.2783 |
| 8345 | 1.6478 / 0.8700 | 2.0511 / 1.8863 |
| 8349 | 1.4712 / 1.3068 | 1.6507 / 1.3468 |
| 8351 | 2.1455 / 1.1276 | 2.5662 / 2.2635 |

All eight prefix images matched the sealed bank and all complete terminal
images matched fresh-bridge replay. Physical distances use the separate
position backend. These are ten-step development outcomes, not the longer
feedback horizon or equal-total-compute baseline required for confirmation.

## Same-action oracle decomposition

The separately [predeclared oracle diagnostic](PC_POINTMAZE_AFFINE_NONPROGRESS_ORACLE_PROTOCOL_20261103.md)
executed **all 36** bank prefixes with their own saved *open-loop*
model-optimized tails. It did not choose actions for the planner. It found
no eligible physical candidate beating both zero and greedy on roots 8343,
8345 or 8351. On exposed root 8349, bank index 4 is a counterexample to the
model's terminal ordering:

| 8349 candidate | Predicted terminal affine score | Physical cp10 distance |
| --- | ---: | ---: |
| Zero, index 0 | 1.3126 | 1.2931 |
| Eligible index 4 | 1.2616 | 1.2606 |
| Model greedy, index 7 | 0.7391 | 1.3142 |

Index 4's physical cp5 distance is 1.6043 versus initial 1.5912, so it also
shows physical position nonprogress under this particular definition.
The model ranked greedy far ahead although that chain's physical endpoint
was worse than index 4 and zero. This identifies a concrete model-score /
physical-outcome ranking failure for these fixed open-loop actions. It does
not prove whether the cause is prediction, affine scoring, RGB velocity
estimation, or their interaction; and the oracle-selected index is not a
valid deployed choice. Replanning from real RGB can choose different tails
and is a separate comparison.

The affine experiment changes both scoring and model-side eligibility
relative to the older proprio-token lookahead; no claim of a controlled
single-variable improvement over it follows. Each root spent 450
two-packet model samples for selection plus the source bank cost and 50
one-packet samples for each executed feedback arm. No strong-baseline
equal-total-budget or independent-root evidence is supplied; training and
normalizer provenance remain unknown. P1 is not passed.

- [Model selection and feedback report](../le-wm-main/results/pc_pointmaze_affine_nonprogress_lookahead_20261103_DEV01.json),
  SHA-256 `a4c1c2a7083195e4951eb26ed559d90af583d6b414eb06125d628c0e9ea17f71`.
- [Physical oracle report](../le-wm-main/results/pc_pointmaze_affine_nonprogress_oracle_20261103.json),
  SHA-256 `6884f717ab6fabc6fa96614c466eb168406c1360b5967041cf46025d44af8a6b`.
- Selection, physical-oracle and related regression checks: **51 passed**.
