# Fixed initial-CEM candidate replay: development result

All 384 sealed candidate plans were executed in the native RGB bridge on four
exposed roots. Each plan used two fixed five-step packets without feedback.
The checkpoint was unchanged and RGB features were computed on host CUDA.
The position-only evaluator replayed the same actions separately. The original
comparison instead replanned its second packet after RGB feedback, so these
open-loop endpoints are not its closed-loop results.

| Comparison within root and arm | Reversed pairs | Total pairs |
| --- | ---: | ---: |
| Model-predicted vs actual RGB task-mask cost | 4,960 | 9,024 |
| Actual RGB cost vs replayed position distance | 2,163 | 9,024 |
| Model-predicted cost vs replayed position distance | 4,683 | 9,024 |

These pairs are correlated: 48 proposals share each root/arm, candidates
derive from sequential CEM generations, and arms share random innovations.
They must not be treated as 9,024 independent observations or used for
confirmatory inference.

The model-selected plans had mean position-distance regret 1.01989 relative
to each arm's best fixed candidate; selecting using actual future RGB cost
would have regret 0.01607. This latter choice is an **oracle using the future**,
not a deployable RGB policy. It shows useful fixed-bank candidates and
motivates checking time-aligned prediction, input history, and normalization;
it does not isolate their causal roles or prove better closed-loop control.
Even actual RGB scoring has 2,163 physical ranking reversals, so metric
alignment is incomplete. Position distance is not native success, and this
new candidate set has not been independently checked against directly
captured live simulator positions.

The action proposals and RGB images match the saved source and image hashes;
saved physical-distance and pairwise arithmetic was independently recomputed
by three passing tests. The tests do not independently rerun all model scores
or prove cross-backend equivalence. All roots were already development roots;
there is no independent P1 result.

- Preregistered local diagnostic:
  `docs/PC_POINTMAZE_CEM_ALL_CANDIDATES_PROTOCOL_20261102.md`
- Runner: `le-wm-main/experiments/a0_pointmaze_cem_all_candidates_20261102.py`
- Result: `le-wm-main/results/a0_pointmaze_cem_all_candidates_20261102_DEV01.json`
  SHA-256 `7dbf437b3f3473c467f6245ac788bd7de07d330f1ac17479b12e9f4b0b1f60fb`
- Arrays SHA-256:
  `2837ff28a0e561f436856db9587617faf1faed73b051e543e43e10f9a2182ab6`
- Validation: `le-wm-main/tests/test_a0_pointmaze_cem_all_candidates_20261102.py`
