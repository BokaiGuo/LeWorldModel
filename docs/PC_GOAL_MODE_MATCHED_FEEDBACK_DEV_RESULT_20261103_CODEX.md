# Goal-mode matched-token RGB feedback: development result

The fixed development protocol ran all five arms on four previously used
PointMaze roots. Each arm made three real-RGB feedback decisions, planned
15 actions at each decision and executed five. All 60 decisions used 132
five-action prediction-token samples, with zero-prefix abstention for the
goal, ridge and random arms. The greedy arm chose by the immediate prefix
score from its own goal-generated bank. Identical bank claims apply at the
same context only; feedback histories may differ after earlier selections.

Post-seal physical replay on MuJoCo 2.1.2.14, terminal distance (lower better):

| Root | Goal mode | Same-bank greedy | Ridge | Random | Direct CEM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.6291 | 0.6291 | 0.3866 | 0.7564 | 0.5906 |
| 1 | 0.4990 | 0.1961 | 0.1468 | 1.3025 | 0.5861 |
| 2 | 0.3683 | 0.3318 | 0.6919 | 0.6005 | 0.5645 |
| 3 | 0.1417 | 0.2358 | 0.4393 | 0.4393 | 0.5206 |
| Mean | 0.4095 | 0.3482 | 0.4162 | 0.7747 | 0.5654 |

The goal arm beat direct CEM on three of four roots, but same-bank greedy
was better on two roots and equal on one; ridge was better on two. These
small, reused roots are not an independent sample supporting a confirmatory
comparison. Goal selection differed from its own instantaneous same-bank
greedy selection on three of twelve decisions. Two goal decisions abstained.
There is no basis here to attribute performance to lookahead or SVD, and
no non-immediate mechanism intervention was performed.

Mean observed decision planning wall seconds were goal 1.2354, greedy 1.2292,
ridge 1.2308, random 1.1837 and CEM 1.1786. Timings are machine-specific,
not FLOPs; encoding and shared model setup are separately recorded. Goal and
greedy each invoked 12 SVDs across 12 decisions, ridge invoked 12 linear
solves, and random and CEM invoked neither. Random candidates did not pay
for an unused Jacobian and used the saved prediction budget on suffix
optimization. Matching prediction tokens does not compensate the other arms
for all goal-mode overhead; `full_cost_certified` remains false.

The development protocol was written after earlier single-root outcomes.
Training membership, normalizer fit exclusion and independent root provenance
are not established. Do not infer success rates or statistical significance
from these four roots. PC2, PC3 and later independent confirmation remain
unpassed; the prior stopped non-immediate PointMaze protocol was not reopened.

The complete decisions, images and action logs have per-root/per-arm names
under `le-wm-main/results/pc_goal_mode_matched_feedback_20261103_codex_*`.
The 20 complete physical traces and source bindings are in
`le-wm-main/results/pc_goal_mode_matched_feedback_20261103_codex_physical.json`.
