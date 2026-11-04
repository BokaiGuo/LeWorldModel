# Exact successor candidate coverage and retention

This post-outcome diagnostic follows
`PC_RGB_SUCCESSOR_EXACT_COVERAGE_ATTRIBUTION_DIAG_PROTOCOL_20261103_CODEX.md`.
It replays the sealed policy's actual normalized actions, including the
float32 conversion used by the predictor, on all 24 exposed wall roots.

| Recorded stage | Total actions or chains | Reached far side (`x > 0`) | Terminal success (`distance < 0.05`) |
|---|---:|---:|---:|
| Generated first prefixes, 5 steps | 48 | 0 | Not applicable |
| Composed prefixes, 10 steps | 96 | 0 | Not applicable |
| Scored complete chains, 15 steps | 3,000 | 6, on 6 roots | 0 |

Some complete samples crossed the wall but none met the success threshold.
Crossing alone does not imply a lower terminal distance: a chain may cross far
from the goal. For all 24 roots the final selected requested action chain was
exactly the lowest-scoring recorded full-chain sample, and the maximum
selected-minus-best score was 0.0. Therefore this diagnostic found **no
instance of a better sampled full chain being dropped by final selection**.
It does not prove intermediate pruning or candidate generation is optimal,
or that the selector would preserve an unobserved useful chain.

The specific observed bottleneck is restricted generated-prefix coverage
and unsuccessful full-chain samples under this budget and action
parameterization. That is an engineering/search diagnosis, not proof that
the method class cannot find solutions. The analytic feasible trajectories
were not injected or used for selection.

Raw counts, lowest-scoring chains, receipts and source hashes are in
`le-wm-main/results/pc_rgb_successor_exact_same_policy_dev_20261103_codex_coverage_attribution_diagnostic.json`.
The coverage, collision and same-chain regression selection passed six tests.
Neither this diagnostic nor the collision association passes PC2 or PC3.
